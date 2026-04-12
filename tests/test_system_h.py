"""Tests for System H: patience-based early exit (PABEE).

System H = System G's per-ramp threshold vector + patience parameter P.
A document exits only when P *consecutive* off-ramps all have entropy below
their respective threshold. P=1 reproduces System G exactly. Inference-only.

These tests exercise:
1. The ``patience`` parameter of ``forward_compacted_early_exit`` (real model,
   CPU, tiny batch — same pattern as ``tests/test_system_f.py``).
2. The sweep driver's ``patience`` plumbing: CSV schema, column values, and
   output directory isolation (mocked GPU — same ``mocked_sweep_env`` fixture
   from ``tests/conftest.py``).
"""

import csv
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
# Patience-based early exit (PABEE) tests
# ---------------------------------------------------------------------------

class TestPatienceEarlyExit:
    """forward_compacted_early_exit must accept a ``patience`` int parameter.
    A document exits only when ``patience`` *consecutive* off-ramps all have
    entropy below their respective per-ramp threshold. P=1 is the current
    (System G) behavior. P>1 delays exits, trading latency for quality.
    """

    def test_patience_1_is_default_behavior(self, model, small_batch):
        """P=1 must produce identical output to the current (no patience) call.
        Backward compatibility proof."""
        with torch.no_grad():
            out_default = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=[0.1, 0.1, 0.1, 0.1, 0.1],
            )
            out_p1 = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=[0.1, 0.1, 0.1, 0.1, 0.1],
                patience=1,
            )
        assert torch.allclose(out_default["scores"], out_p1["scores"], atol=1e-6)
        assert torch.equal(out_default["exit_layer"], out_p1["exit_layer"])
        assert out_default["exit_counts"] == out_p1["exit_counts"]

    def test_patience_2_delays_exit(self, model, small_batch):
        """With thresholds [1.0, 1.0, 0.0, 0.0, 0.0] every doc has entropy < 1.0
        at ramps 0 and 1. P=1 exits all at ramp 0; P=2 exits all at ramp 1
        (first ramp where 2 consecutive below-threshold ramps have been seen)."""
        batch_size = small_batch["input_ids"].shape[0]
        thresholds = [1.0, 1.0, 0.0, 0.0, 0.0]
        with torch.no_grad():
            out_p1 = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=thresholds,
                patience=1,
            )
            out_p2 = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=thresholds,
                patience=2,
            )
        # P=1: all exit at ramp 0
        assert out_p1["exit_counts"][0] == batch_size
        # P=2: all exit at ramp 1 (ramp 0 = count 1, ramp 1 = count 2 >= 2)
        assert out_p2["exit_counts"][0] == 0, (
            f"P=2 should not exit at ramp 0. exit_counts={out_p2['exit_counts']}"
        )
        assert out_p2["exit_counts"][1] == batch_size, (
            f"P=2 should exit all at ramp 1. exit_counts={out_p2['exit_counts']}"
        )

    def test_patience_3_delays_further(self, model, small_batch):
        """With thresholds [1.0, 1.0, 1.0, 0.0, 0.0], P=3 exits all at ramp 2
        (ramps 0, 1, 2 all below threshold = 3 consecutive)."""
        batch_size = small_batch["input_ids"].shape[0]
        thresholds = [1.0, 1.0, 1.0, 0.0, 0.0]
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=thresholds,
                patience=3,
            )
        assert out["exit_counts"][0] == 0
        assert out["exit_counts"][1] == 0
        assert out["exit_counts"][2] == batch_size, (
            f"P=3 should exit all at ramp 2. exit_counts={out['exit_counts']}"
        )

    def test_patience_resets_on_above_threshold(self, model, small_batch):
        """With thresholds [1.0, 0.0, 1.0, 1.0, 0.0] and P=2:
        ramp 0 below (count=1), ramp 1 above (count resets to 0),
        ramp 2 below (count=1), ramp 3 below (count=2 -> exit at ramp 3)."""
        batch_size = small_batch["input_ids"].shape[0]
        thresholds = [1.0, 0.0, 1.0, 1.0, 0.0]
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=thresholds,
                patience=2,
            )
        assert out["exit_counts"][0] == 0
        assert out["exit_counts"][1] == 0
        assert out["exit_counts"][2] == 0
        assert out["exit_counts"][3] == batch_size, (
            f"P=2 with reset should exit all at ramp 3. exit_counts={out['exit_counts']}"
        )

    def test_patience_exceeds_ramps_goes_to_final(self, model, small_batch):
        """P=6 is impossible to satisfy with only 5 off-ramps -- all docs must
        reach the final layer (exit_layer=5) regardless of thresholds."""
        batch_size = small_batch["input_ids"].shape[0]
        thresholds = [1.0, 1.0, 1.0, 1.0, 1.0]
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=thresholds,
                patience=6,
            )
        for i in range(5):
            assert out["exit_counts"][i] == 0, (
                f"P=6 should not exit at ramp {i}. exit_counts={out['exit_counts']}"
            )
        assert out["exit_counts"][5] == batch_size, (
            f"P=6 should send all to final. exit_counts={out['exit_counts']}"
        )

    def test_patience_invalid_values(self, model, small_batch):
        """P=0, P=-1, P=1.5, P='2' must raise ValueError or TypeError."""
        for bad_val in (0, -1):
            with pytest.raises(ValueError):
                model.forward_compacted_early_exit(
                    small_batch["input_ids"],
                    small_batch["attention_mask"],
                    small_batch["token_type_ids"],
                    entropy_threshold=0.1,
                    patience=bad_val,
                )
        for bad_type in (1.5, "2", True):
            with pytest.raises(TypeError):
                model.forward_compacted_early_exit(
                    small_batch["input_ids"],
                    small_batch["attention_mask"],
                    small_batch["token_type_ids"],
                    entropy_threshold=0.1,
                    patience=bad_type,
                )


# ---------------------------------------------------------------------------
# Sweep driver — patience plumbing tests
# ---------------------------------------------------------------------------
#
# These guard the contract between the SLURM script's --patience flag and the
# CSV that lands on disk. The mocked_sweep_env fixture (tests/conftest.py)
# replaces all GPU-heavy dependencies so these run anywhere.

class TestSweepDriverPatience:
    """Sweep driver must accept ``patience``, record it in every CSV row,
    and preserve the existing column schema so System F/G CSVs stay
    comparable."""

    def test_default_patience_csv_schema_unchanged(self, mocked_sweep_env):
        """With default kwargs (no patience arg), the CSV header must include
        a 'patience' column AND all original columns in the same order.
        Every row must have patience=1."""
        from src.inference import run_per_ramp_threshold_sweep

        csv_path = run_per_ramp_threshold_sweep(
            task_id=0,
            num_tasks=4,
            grid_values=[0.01, 0.5],
            output_dir=str(mocked_sweep_env),
        )

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            header = reader.fieldnames

        # Original columns must still be present
        for col in ["t0", "t1", "t2", "t3", "t4", "mrr10",
                     "mean_batch_latency_ms", "exit_count_0", "exit_count_1",
                     "exit_count_2", "exit_count_3", "exit_count_4",
                     "exit_count_5"]:
            assert col in header, f"Missing original column '{col}' in header"

        # patience column must exist
        assert "patience" in header, "CSV must include 'patience' column"

        # Every row must have patience=1 (default)
        for row in rows:
            assert row["patience"] == "1", (
                f"Default patience should be 1, got {row['patience']}"
            )

    def test_patience_2_recorded_in_csv(self, mocked_sweep_env):
        """With patience=2, every CSV row must have patience=2."""
        from src.inference import run_per_ramp_threshold_sweep

        csv_path = run_per_ramp_threshold_sweep(
            task_id=0,
            num_tasks=4,
            grid_values=[0.01, 0.5],
            output_dir=str(mocked_sweep_env),
            patience=2,
        )

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 8, f"Expected 8 rows, got {len(rows)}"
        for row in rows:
            assert row["patience"] == "2", (
                f"patience=2 not recorded. Got {row['patience']}"
            )

    def test_system_h_sweep_layout_p2(self, mocked_sweep_env):
        """System H P=2 sweep must write CSVs to system_h_sweep_results/
        with the system_h_p2 prefix, and must NOT create System F or G dirs."""
        from src.constants import DEFAULT_PER_RAMP_GRID
        from src.inference import run_per_ramp_threshold_sweep

        produced = []
        for task_id in range(49):
            csv_path = run_per_ramp_threshold_sweep(
                task_id=task_id,
                num_tasks=49,
                grid_values=DEFAULT_PER_RAMP_GRID,
                output_dir=str(mocked_sweep_env),
                sweep_subdir="system_h_sweep_results",
                csv_prefix="system_h_p2",
                patience=2,
            )
            produced.append(csv_path)

        # All 49 CSVs in the right directory
        expected_dir = os.path.join(str(mocked_sweep_env), "system_h_sweep_results")
        for p in produced:
            assert os.path.dirname(p) == expected_dir
            assert os.path.basename(p).startswith("system_h_p2_t0=")
            assert os.path.isfile(p)

        # 49 unique cells
        assert len(set(produced)) == 49

        # Expected filenames cover full grid
        expected_basenames = {
            f"system_h_p2_t0={t0}_t1={t1}.csv"
            for t0 in DEFAULT_PER_RAMP_GRID
            for t1 in DEFAULT_PER_RAMP_GRID
        }
        produced_basenames = {os.path.basename(p) for p in produced}
        assert produced_basenames == expected_basenames

        # Must not touch System F or G directories
        assert not os.path.isdir(
            os.path.join(str(mocked_sweep_env), "system_f_sweep_results")
        ), "System H leaked into System F dir"
        assert not os.path.isdir(
            os.path.join(str(mocked_sweep_env), "system_g_sweep_results")
        ), "System H leaked into System G dir"
