"""Tests for System F: per-ramp entropy threshold vector support.

System F = System D α=0.5 weights + per-ramp entropy threshold vector
(length 5, one float per off-ramp) instead of a single scalar threshold.

The underlying inference function is the same compacted early-exit forward
pass used by System C / System D. These tests cover the new vector-input
capability that System F relies on. The existing scalar-input behavior is
covered by tests in tests/test_system_c.py and must remain unchanged.
"""

import os

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    from src.model import EarlyExitCrossEncoder
    m = EarlyExitCrossEncoder()
    m.eval()
    return m


@pytest.fixture(scope="module")
def small_batch(model):
    """Return a small tokenized batch (batch_size=4) for quick unit tests."""
    tokenizer = model.tokenizer
    queries = ["what is python", "best gpu", "neural network", "early exit"]
    passages = [
        "Python is a programming language.",
        "The RTX 4090 is a high-end GPU.",
        "Neural networks are inspired by the brain.",
        "Early exit reduces computation.",
    ]
    encoded = tokenizer(
        queries,
        passages,
        max_length=32,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return encoded


# ---------------------------------------------------------------------------
# Per-ramp threshold vector tests
# ---------------------------------------------------------------------------

class TestForwardCompactedEarlyExitPerRampThresholds:
    """forward_compacted_early_exit must accept either a single float
    (broadcast to all 5 off-ramps, current behavior) or a length-5 sequence
    of floats (one threshold per off-ramp).
    """

    def test_uniform_sequence_equivalent_to_float(self, model, small_batch):
        """Passing [t]*5 must produce results identical to passing the float t."""
        with torch.no_grad():
            out_float = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
            out_seq = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=[0.1, 0.1, 0.1, 0.1, 0.1],
            )
        assert torch.allclose(out_float["scores"], out_seq["scores"], atol=1e-6), (
            "Uniform sequence [0.1]*5 must give identical scores to float 0.1"
        )
        assert torch.equal(out_float["exit_layer"], out_seq["exit_layer"]), (
            f"Uniform sequence must give identical exit_layer. "
            f"float: {out_float['exit_layer'].tolist()}, "
            f"seq: {out_seq['exit_layer'].tolist()}"
        )
        assert out_float["exit_counts"] == out_seq["exit_counts"], (
            f"Uniform sequence must give identical exit_counts. "
            f"float: {out_float['exit_counts']}, "
            f"seq: {out_seq['exit_counts']}"
        )

    def test_per_ramp_thresholds_take_effect(self, model, small_batch):
        """Different threshold vectors must produce different exit behavior.

        With [1.0, 0.0, 0.0, 0.0, 0.0] every doc must exit at ramp 0
        (max binary entropy ln(2) ≈ 0.693 < 1.0).
        With [0.0, 1.0, 0.0, 0.0, 0.0] no doc may exit at ramp 0 (entropy
        is never < 0), but every doc must exit at ramp 1 instead.
        """
        batch_size = small_batch["input_ids"].shape[0]
        with torch.no_grad():
            out_loose0 = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=[1.0, 0.0, 0.0, 0.0, 0.0],
            )
            out_loose1 = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=[0.0, 1.0, 0.0, 0.0, 0.0],
            )
        assert out_loose0["exit_counts"][0] == batch_size, (
            f"With t=[1.0, 0, 0, 0, 0], all docs must exit at ramp 0. "
            f"exit_counts={out_loose0['exit_counts']}"
        )
        assert sum(out_loose0["exit_counts"][1:]) == 0
        assert out_loose1["exit_counts"][0] == 0, (
            f"With t=[0, 1.0, 0, 0, 0], no docs may exit at ramp 0. "
            f"exit_counts={out_loose1['exit_counts']}"
        )
        assert out_loose1["exit_counts"][1] == batch_size, (
            f"With t=[0, 1.0, 0, 0, 0], all docs must exit at ramp 1. "
            f"exit_counts={out_loose1['exit_counts']}"
        )

    def test_wrong_length_sequence_raises_value_error(self, model, small_batch):
        """A sequence of length != 5 must raise ValueError."""
        for bad_len_vec in ([0.1] * 4, [0.1] * 6, [], [0.1]):
            with pytest.raises(ValueError):
                model.forward_compacted_early_exit(
                    small_batch["input_ids"],
                    small_batch["attention_mask"],
                    small_batch["token_type_ids"],
                    entropy_threshold=bad_len_vec,
                )

    def test_invalid_type_raises_type_error(self, model, small_batch):
        """Non-float, non-sequence types must raise TypeError.

        bool is included explicitly: it is an int subclass and would otherwise
        silently coerce True/False into 1.0/0.0 — almost certainly a bug.
        """
        for bad in ("0.1", {"a": 0.1}, None, True, False):
            with pytest.raises(TypeError):
                model.forward_compacted_early_exit(
                    small_batch["input_ids"],
                    small_batch["attention_mask"],
                    small_batch["token_type_ids"],
                    entropy_threshold=bad,
                )


# ---------------------------------------------------------------------------
# run_per_ramp_threshold_sweep — sweep driver tests
# ---------------------------------------------------------------------------
#
# These tests guard the contract of the System F sweep driver itself
# (partitioning math, CSV schema, input validation). They are designed to
# catch silent bugs that would otherwise only surface during post-sweep
# manual analysis — e.g., a partitioning typo that skips a (t0, t1) cell, a
# header rename that breaks downstream tooling, or a validation guard regression.
# All GPU-heavy dependencies are mocked so the tests run on any machine.

class TestRunPerRampThresholdSweep:

    @pytest.fixture
    def mocked_sweep_env(self, monkeypatch, tmp_path):
        """Patch GPU-heavy dependencies of run_per_ramp_threshold_sweep so tests
        only exercise partitioning, validation, and CSV writing. Returns the
        tmp_path that should be passed as ``output_dir``.
        """
        import src.inference as inf

        class _FakeStateDictHolder:
            def load_state_dict(self, _state):
                pass

        class _FakeModel:
            def __init__(self):
                self.backbone = _FakeStateDictHolder()
                self.offramps = _FakeStateDictHolder()

            def to(self, _device):
                return self

            def eval(self):
                return self

            def forward_compacted_early_exit(self, *_a, **_k):
                return {
                    "scores": torch.zeros(1),
                    "exit_counts": [1, 0, 0, 0, 0, 0],
                    "exit_layer": torch.zeros(1, dtype=torch.long),
                }

        monkeypatch.setattr(inf, "EarlyExitCrossEncoder", _FakeModel)
        monkeypatch.setattr(
            torch, "load", lambda *a, **k: {"backbone": {}, "offramps": {}}
        )

        fake_data = {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.zeros(2, 4, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 4, dtype=torch.long),
            "qids": [0, 0],
            "labels": [1, 0],
        }
        monkeypatch.setattr(inf, "load_tokenized_data", lambda *a, **k: fake_data)

        class _FakeBatchRunner:
            def __init__(self, *a, **k):
                pass

            def warmup(self, *a, **k):
                pass

            def run_with_timing(self, *a, **k):
                fake_out = {
                    "scores": torch.zeros(2),
                    "exit_counts": [2, 0, 0, 0, 0, 0],
                }
                return [fake_out], [1.0]

        monkeypatch.setattr(inf, "BatchRunner", _FakeBatchRunner)
        monkeypatch.setattr(inf, "compute_mrr_at_k", lambda *a, **k: 0.0)

        return tmp_path

    def test_per_ramp_sweep_partitioning_covers_full_grid(self, mocked_sweep_env):
        """task_id 0..N*N-1 must produce N*N unique (t0, t1) cells covering the
        full Cartesian product of grid_values. Verified with the default 7-value
        grid (49 cells) — the same partitioning the cluster sweep uses.
        """
        from src.constants import DEFAULT_PER_RAMP_GRID
        from src.inference import run_per_ramp_threshold_sweep

        produced = []
        for task_id in range(49):
            csv_path = run_per_ramp_threshold_sweep(
                task_id=task_id,
                num_tasks=49,
                grid_values=DEFAULT_PER_RAMP_GRID,
                output_dir=str(mocked_sweep_env),
            )
            produced.append(os.path.basename(csv_path))

        expected = {
            f"system_f_t0={t0}_t1={t1}.csv"
            for t0 in DEFAULT_PER_RAMP_GRID
            for t1 in DEFAULT_PER_RAMP_GRID
        }
        assert set(produced) == expected, (
            f"Partitioning did not cover full grid. "
            f"Missing: {expected - set(produced)}. "
            f"Extra: {set(produced) - expected}."
        )
        assert len(produced) == 49
        assert len(set(produced)) == 49, "Some (t0, t1) cells were produced twice"

    def test_per_ramp_sweep_csv_schema(self, mocked_sweep_env):
        """The CSV header written by the driver must match the documented schema.

        Uses a 2-value grid (4 tasks, 8 inner rows) so the test runs fast while
        still exercising the full CSV-writing path.
        """
        from src.inference import run_per_ramp_threshold_sweep

        csv_path = run_per_ramp_threshold_sweep(
            task_id=0,
            num_tasks=4,
            grid_values=[0.01, 0.5],
            output_dir=str(mocked_sweep_env),
        )

        expected_header = [
            "t0", "t1", "t2", "t3", "t4",
            "mrr10", "mean_batch_latency_ms",
            "exit_count_0", "exit_count_1", "exit_count_2",
            "exit_count_3", "exit_count_4", "exit_count_5",
        ]
        with open(csv_path, "r") as f:
            header = f.readline().strip().split(",")
            data_rows = [line for line in f.read().splitlines() if line]

        assert header == expected_header, (
            f"CSV header drift. Expected {expected_header}, got {header}"
        )
        # 2**3 = 8 inner (t2, t3, t4) configs
        assert len(data_rows) == 8, (
            f"Expected 8 data rows for grid_size=2, got {len(data_rows)}"
        )

    def test_per_ramp_sweep_validates_inputs(self):
        """Driver must reject mismatched num_tasks and out-of-range task_id
        before any GPU work — these guards run before model load, so no
        mocking is required.
        """
        from src.inference import run_per_ramp_threshold_sweep

        with pytest.raises(ValueError, match="num_tasks must be 49"):
            run_per_ramp_threshold_sweep(task_id=0, num_tasks=10)

        with pytest.raises(ValueError, match="task_id must be in"):
            run_per_ramp_threshold_sweep(task_id=49, num_tasks=49)

        with pytest.raises(ValueError, match="task_id must be in"):
            run_per_ramp_threshold_sweep(task_id=-1, num_tasks=49)
