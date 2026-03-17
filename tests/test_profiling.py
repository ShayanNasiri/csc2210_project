"""Tests for Phase 7: profiling sweep and final evaluation.

Import/existence tests run immediately.
CSV-dependent tests skip until full_sweep.csv is produced by the SLURM job.
"""

import inspect
import os

import pytest


CSV_PATH = "results/full_sweep.csv"
CSV_EXISTS = os.path.exists(CSV_PATH)

EXPECTED_SYSTEMS = {"baseline_a", "baseline_b", "system_c"}
EXPECTED_BATCH_SIZES = {32, 64, 128, 256, 512}
CORE_COLUMNS = {"system", "batch_size", "threshold", "mrr10", "mean_latency_ms", "std_latency_ms"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sweep_df():
    """Load full_sweep.csv — only used in CSV-dependent tests."""
    import pandas as pd
    return pd.read_csv(CSV_PATH)


# ---------------------------------------------------------------------------
# Import and existence tests (always run)
# ---------------------------------------------------------------------------

class TestImportsAndFiles:

    def test_run_full_sweep_importable(self):
        from src.inference import run_full_sweep  # noqa: F401

    def test_full_sweep_signature(self):
        from src.inference import run_full_sweep
        sig = inspect.signature(run_full_sweep)
        params = set(sig.parameters.keys())
        assert "tokenized_path" in params, "run_full_sweep must accept tokenized_path"
        assert "systems" in params, "run_full_sweep must accept systems"
        assert "batch_sizes" in params, "run_full_sweep must accept batch_sizes"
        assert "thresholds" in params, "run_full_sweep must accept thresholds"
        assert "output_dir" in params, "run_full_sweep must accept output_dir"

    def test_ncu_microbench_exists(self):
        assert os.path.exists("scripts/ncu_microbench.py"), (
            "scripts/ncu_microbench.py must exist"
        )

    def test_run_profiling_sh_exists(self):
        assert os.path.exists("scripts/run_profiling.sh"), (
            "scripts/run_profiling.sh must exist"
        )

    def test_run_ncu_sh_exists(self):
        assert os.path.exists("scripts/run_ncu.sh"), (
            "scripts/run_ncu.sh must exist"
        )

    def test_plot_pareto_script_exists(self):
        assert os.path.exists("scripts/plot_pareto.py"), (
            "scripts/plot_pareto.py must exist"
        )


# ---------------------------------------------------------------------------
# CSV-dependent tests (skip until full_sweep.csv exists)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not CSV_EXISTS, reason="full_sweep.csv not found — run sbatch scripts/run_profiling.sh first")
class TestFullSweepCSV:

    def test_sweep_csv_schema(self, sweep_df):
        actual = set(sweep_df.columns)
        missing = CORE_COLUMNS - actual
        assert not missing, (
            f"full_sweep.csv missing columns: {missing}. Got: {actual}"
        )

    def test_sweep_all_systems_present(self, sweep_df):
        actual = set(sweep_df["system"].unique())
        missing = EXPECTED_SYSTEMS - actual
        assert not missing, (
            f"Missing systems in sweep: {missing}. Got: {actual}"
        )

    def test_sweep_all_batch_sizes(self, sweep_df):
        actual = set(sweep_df["batch_size"].unique())
        missing = EXPECTED_BATCH_SIZES - actual
        assert not missing, (
            f"Missing batch sizes in sweep: {missing}. Got: {actual}"
        )

    def test_sweep_mrr_range(self, sweep_df):
        out_of_range = sweep_df[~sweep_df["mrr10"].between(0.0, 1.0, inclusive="neither")]
        assert len(out_of_range) == 0, (
            f"MRR@10 values out of (0, 1] range:\n{out_of_range[['system', 'batch_size', 'threshold', 'mrr10']]}"
        )

    def test_sweep_latency_positive(self, sweep_df):
        bad_mean = sweep_df[sweep_df["mean_latency_ms"] <= 0]
        assert len(bad_mean) == 0, (
            f"Non-positive mean_latency_ms found:\n{bad_mean[['system', 'batch_size', 'threshold', 'mean_latency_ms']]}"
        )
        bad_std = sweep_df[sweep_df["std_latency_ms"] < 0]
        assert len(bad_std) == 0, (
            f"Negative std_latency_ms found:\n{bad_std[['system', 'batch_size', 'threshold', 'std_latency_ms']]}"
        )

    def test_system_c_faster_than_b(self, sweep_df):
        """For each (threshold, batch_size) pair, System C must be faster than Baseline B."""
        b_rows = sweep_df[sweep_df["system"] == "baseline_b"].set_index(["batch_size", "threshold"])
        c_rows = sweep_df[sweep_df["system"] == "system_c"].set_index(["batch_size", "threshold"])

        common = b_rows.index.intersection(c_rows.index)
        assert len(common) > 0, "No matching (batch_size, threshold) pairs between baseline_b and system_c"

        failures = []
        for idx in common:
            b_lat = b_rows.loc[idx, "mean_latency_ms"]
            c_lat = c_rows.loc[idx, "mean_latency_ms"]
            if c_lat >= b_lat:
                failures.append(
                    f"batch_size={idx[0]}, threshold={idx[1]}: "
                    f"System C ({c_lat:.2f}ms) >= Baseline B ({b_lat:.2f}ms)"
                )

        assert not failures, "System C must be faster than Baseline B:\n" + "\n".join(failures)
