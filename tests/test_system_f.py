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
    """Sweep-driver contract tests. The ``mocked_sweep_env`` fixture lives in
    tests/conftest.py because tests/test_system_g.py reuses it for its
    end-to-end layout test."""

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
            "patience",
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

    # ---------------------------------------------------------------------
    # sweep_subdir / csv_prefix parameterization
    # ---------------------------------------------------------------------
    #
    # The sweep driver must accept ``sweep_subdir`` and ``csv_prefix`` kwargs
    # so other weight families (e.g. the System E β=1.0 weights used by the
    # planned System G sweep) can run the same grid without colliding with
    # System F's already-committed CSVs at
    # ``results/system_f_sweep_results/system_f_t0=*.csv``.
    #
    # Default values must reproduce the System F layout exactly so re-running
    # the System F sweep produces a byte-identical directory tree to what is
    # currently committed in the repo.

    def test_per_ramp_sweep_default_subdir_and_prefix(self, mocked_sweep_env):
        """Default kwargs must produce the System F layout exactly:
        ``<output_dir>/system_f_sweep_results/system_f_t0=<t0>_t1=<t1>.csv``.
        Guards against accidental drift in the committed System F path.
        """
        from src.inference import run_per_ramp_threshold_sweep

        csv_path = run_per_ramp_threshold_sweep(
            task_id=0,
            num_tasks=4,
            grid_values=[0.01, 0.5],
            output_dir=str(mocked_sweep_env),
        )

        expected_subdir = os.path.join(str(mocked_sweep_env), "system_f_sweep_results")
        assert os.path.dirname(csv_path) == expected_subdir, (
            f"Default subdir drift. Expected {expected_subdir}, "
            f"got {os.path.dirname(csv_path)}"
        )
        assert os.path.basename(csv_path) == "system_f_t0=0.01_t1=0.01.csv", (
            f"Default csv_prefix drift. Expected system_f_t0=0.01_t1=0.01.csv, "
            f"got {os.path.basename(csv_path)}"
        )
        assert os.path.isfile(csv_path), f"CSV not written to {csv_path}"

    def test_per_ramp_sweep_custom_subdir_isolated_from_default(self, mocked_sweep_env):
        """A custom ``sweep_subdir`` must place the CSV under that directory
        and must NOT touch the default System F subdir as a side effect."""
        from src.inference import run_per_ramp_threshold_sweep

        csv_path = run_per_ramp_threshold_sweep(
            task_id=0,
            num_tasks=4,
            grid_values=[0.01, 0.5],
            output_dir=str(mocked_sweep_env),
            sweep_subdir="some_other_subdir",
        )

        expected_subdir = os.path.join(str(mocked_sweep_env), "some_other_subdir")
        assert os.path.dirname(csv_path) == expected_subdir, (
            f"Custom sweep_subdir not honored. Expected {expected_subdir}, "
            f"got {os.path.dirname(csv_path)}"
        )
        assert os.path.isfile(csv_path)
        # Default csv_prefix is still system_f when only subdir is overridden
        assert os.path.basename(csv_path).startswith("system_f_t0="), (
            "csv_prefix should remain 'system_f' when only sweep_subdir is overridden"
        )
        # The default System F subdir must NOT have been created
        assert not os.path.isdir(
            os.path.join(str(mocked_sweep_env), "system_f_sweep_results")
        ), "Custom sweep_subdir leaked into default system_f_sweep_results dir"

    def test_per_ramp_sweep_custom_csv_prefix(self, mocked_sweep_env):
        """A custom ``csv_prefix`` must change the filename stem while leaving
        the default subdir alone."""
        from src.inference import run_per_ramp_threshold_sweep

        csv_path = run_per_ramp_threshold_sweep(
            task_id=2,
            num_tasks=4,
            grid_values=[0.01, 0.5],
            output_dir=str(mocked_sweep_env),
            csv_prefix="my_custom_prefix",
        )

        # task_id=2 with grid [0.01, 0.5] -> t0_idx=1, t1_idx=0 -> (0.5, 0.01)
        assert os.path.basename(csv_path) == "my_custom_prefix_t0=0.5_t1=0.01.csv", (
            f"Custom csv_prefix not honored. Got {os.path.basename(csv_path)}"
        )
        # Subdir defaults to system_f_sweep_results when only csv_prefix is overridden
        assert os.path.basename(os.path.dirname(csv_path)) == "system_f_sweep_results"
        assert os.path.isfile(csv_path)


# ---------------------------------------------------------------------------
# run_system_f — single-operating-point runner tests
# ---------------------------------------------------------------------------
#
# System F at the winning per-ramp threshold vector is a LEGITIMATE named
# operating point for the final eval, not a sweep cell. The single-point
# runner below evaluates one (thresholds, patience=1, weights) configuration
# and writes a scalar result JSON matching the Baseline A / System D schema.

class TestRunSystemF:
    def test_importable(self):
        from src.inference import run_system_f  # noqa: F401

    def test_signature(self):
        import inspect
        from src.inference import run_system_f
        sig = inspect.signature(run_system_f)
        for name in ("tokenized_path", "batch_size", "thresholds",
                     "output_dir", "weights_path", "results_tag"):
            assert name in sig.parameters, f"missing param {name}"

    def test_wrong_length_thresholds_raises(self, mocked_sweep_env):
        from src.inference import run_system_f
        for bad in ([0.1] * 4, [0.1] * 6, []):
            with pytest.raises(ValueError, match="length 5"):
                run_system_f(
                    thresholds=bad,
                    output_dir=str(mocked_sweep_env),
                    weights_path="irrelevant.pt",
                )

    def test_writes_json_with_expected_schema(self, mocked_sweep_env):
        import json
        from src.inference import run_system_f
        result = run_system_f(
            thresholds=[0.001, 0.01, 0.3, 0.5, 0.5],
            output_dir=str(mocked_sweep_env),
            weights_path="irrelevant.pt",
            results_tag="test_final_",
        )
        assert isinstance(result, dict)
        assert result["system"] == "system_f"
        assert result["thresholds"] == [0.001, 0.01, 0.3, 0.5, 0.5]
        assert result["patience"] == 1
        assert "mrr10" in result and "mean_batch_latency_ms" in result
        assert "exit_counts" in result

        json_path = os.path.join(str(mocked_sweep_env), "test_final_system_f_results.json")
        assert os.path.isfile(json_path), f"expected JSON at {json_path}"
        with open(json_path) as f:
            on_disk = json.load(f)
        # Match System D/E convention: JSON contains a list of result dicts
        assert isinstance(on_disk, list) and len(on_disk) == 1
        assert on_disk[0]["system"] == "system_f"

    def test_cli_dispatch_routes_to_run_system_f(self):
        import inspect
        import src.inference as inf
        src = inspect.getsource(inf)
        branch = src.split('elif args.system == "system_f":')[1].split("elif args.system ==")[0]
        assert "run_system_f(" in branch
        assert "thresholds=args.thresholds" in branch or "thresholds=" in branch
        assert "weights_path=args.weights_path" in branch
