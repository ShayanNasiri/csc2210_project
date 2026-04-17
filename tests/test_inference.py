"""Tests for src/inference.py — inference drivers, CLI, and helper functions."""

import inspect
import os
import subprocess
import tempfile

import pytest
import torch

from src.constants import NUM_OFFRAMPS, DEFAULT_ENTROPY_THRESHOLDS


# ---- _exit_counts_to_pct tests ----

class TestExitCountsToPct:
    """Tests for the _exit_counts_to_pct helper function."""

    def test_importable(self):
        from src.inference import _exit_counts_to_pct

    def test_uniform_distribution(self):
        from src.inference import _exit_counts_to_pct
        # 6 layers, 10 each = 60 total
        exit_counts = [10, 10, 10, 10, 10, 10]
        total = 60
        pct = _exit_counts_to_pct(exit_counts, total)

        for i in range(NUM_OFFRAMPS):
            assert abs(pct[f"pct_exit_layer{i}"] - 1 / 6) < 1e-9
        assert abs(pct["pct_exit_final"] - 1 / 6) < 1e-9

    def test_all_exit_at_ramp0(self):
        from src.inference import _exit_counts_to_pct
        exit_counts = [100, 0, 0, 0, 0, 0]
        pct = _exit_counts_to_pct(exit_counts, 100)

        assert pct["pct_exit_layer0"] == 1.0
        for i in range(1, NUM_OFFRAMPS):
            assert pct[f"pct_exit_layer{i}"] == 0.0
        assert pct["pct_exit_final"] == 0.0

    def test_all_exit_at_final(self):
        from src.inference import _exit_counts_to_pct
        exit_counts = [0, 0, 0, 0, 0, 50]
        pct = _exit_counts_to_pct(exit_counts, 50)

        for i in range(NUM_OFFRAMPS):
            assert pct[f"pct_exit_layer{i}"] == 0.0
        assert pct["pct_exit_final"] == 1.0

    def test_zero_total_returns_zeros(self):
        from src.inference import _exit_counts_to_pct
        exit_counts = [0, 0, 0, 0, 0, 0]
        pct = _exit_counts_to_pct(exit_counts, 0)

        for i in range(NUM_OFFRAMPS):
            assert pct[f"pct_exit_layer{i}"] == 0.0
        assert pct["pct_exit_final"] == 0.0

    def test_output_keys(self):
        from src.inference import _exit_counts_to_pct
        pct = _exit_counts_to_pct([10, 20, 30, 0, 0, 40], 100)

        expected_keys = {f"pct_exit_layer{i}" for i in range(NUM_OFFRAMPS)} | {"pct_exit_final"}
        assert set(pct.keys()) == expected_keys

    def test_percentages_sum_to_one(self):
        from src.inference import _exit_counts_to_pct
        exit_counts = [5, 15, 10, 20, 30, 20]
        total = sum(exit_counts)
        pct = _exit_counts_to_pct(exit_counts, total)

        total_pct = sum(pct.values())
        assert abs(total_pct - 1.0) < 1e-9


# ---- run_* function signature tests ----

class TestFunctionSignatures:
    """Verify all run_* functions have consistent signatures."""

    @pytest.fixture(params=["run_baseline_a", "run_baseline_b", "run_system_c", "run_system_d"])
    def func(self, request):
        import src.inference as mod
        return getattr(mod, request.param)

    def test_has_tokenized_path_param(self, func):
        sig = inspect.signature(func)
        assert "tokenized_path" in sig.parameters

    def test_has_batch_size_param(self, func):
        sig = inspect.signature(func)
        assert "batch_size" in sig.parameters

    def test_has_output_dir_param(self, func):
        sig = inspect.signature(func)
        assert "output_dir" in sig.parameters


class TestThresholdFunctions:
    """run_baseline_b, run_system_c, and run_system_d accept thresholds."""

    @pytest.fixture(params=["run_baseline_b", "run_system_c", "run_system_d"])
    def func(self, request):
        import src.inference as mod
        return getattr(mod, request.param)

    def test_has_thresholds_param(self, func):
        sig = inspect.signature(func)
        assert "thresholds" in sig.parameters

    def test_thresholds_default_is_none(self, func):
        sig = inspect.signature(func)
        assert sig.parameters["thresholds"].default is None


# ---- run_full_sweep tests ----

class TestRunFullSweep:
    """Tests for run_full_sweep function."""

    def test_importable(self):
        from src.inference import run_full_sweep

    def test_signature(self):
        from src.inference import run_full_sweep
        sig = inspect.signature(run_full_sweep)
        params = set(sig.parameters.keys())
        for name in ("tokenized_path", "systems", "batch_sizes", "thresholds", "output_dir"):
            assert name in params

    def test_default_sweep_systems(self):
        from src.inference import DEFAULT_SWEEP_SYSTEMS
        assert "baseline_a" in DEFAULT_SWEEP_SYSTEMS
        assert "baseline_b" in DEFAULT_SWEEP_SYSTEMS
        assert "system_c" in DEFAULT_SWEEP_SYSTEMS

    def test_default_sweep_batch_sizes(self):
        from src.inference import DEFAULT_SWEEP_BATCH_SIZES
        assert all(isinstance(b, int) for b in DEFAULT_SWEEP_BATCH_SIZES)
        assert len(DEFAULT_SWEEP_BATCH_SIZES) >= 3


# ---- CLI argument parsing tests ----

class TestCLI:
    """Tests for inference.py CLI argument parsing."""

    def test_help_exits_zero(self):
        result = subprocess.run(
            ["python", "-m", "src.inference", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_baseline_a_help(self):
        result = subprocess.run(
            ["python", "-m", "src.inference", "--system", "baseline_a", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_system_d_help(self):
        result = subprocess.run(
            ["python", "-m", "src.inference", "--system", "system_d", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_full_sweep_help(self):
        result = subprocess.run(
            ["python", "-m", "src.inference", "--system", "full_sweep", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_accepts_batch_size(self):
        result = subprocess.run(
            ["python", "-m", "src.inference", "--batch_size", "32", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0

    def test_cli_accepts_data_path(self):
        result = subprocess.run(
            ["python", "-m", "src.inference", "--data_path", "foo.pt", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0


# ---- System D weight loading tests ----

class TestSystemDWeightLoading:
    """Tests for System D joint weight loading pattern."""

    def test_loads_backbone_and_offramps(self):
        """run_system_d source must load both backbone and offramps from joint weights."""
        source = inspect.getsource(__import__("src.inference", fromlist=["run_system_d"]).run_system_d)
        assert 'state["backbone"]' in source or "backbone" in source
        assert 'state["offramps"]' in source or "offramps" in source

    def test_uses_forward_compacted_early_exit(self):
        """System D must use Triton-compacted forward pass."""
        source = inspect.getsource(__import__("src.inference", fromlist=["run_system_d"]).run_system_d)
        assert "forward_compacted_early_exit" in source

    def test_saves_system_d_results(self):
        """System D must save results as system_d_results.json."""
        source = inspect.getsource(__import__("src.inference", fromlist=["run_system_d"]).run_system_d)
        assert "system_d_results.json" in source

    def test_system_name_is_system_d(self):
        """Result dict must have system='system_d'."""
        source = inspect.getsource(__import__("src.inference", fromlist=["run_system_d"]).run_system_d)
        assert '"system_d"' in source

    def test_run_system_d_has_weights_path_param(self):
        """run_system_d must accept a weights_path parameter (mirrors System E for alpha sweep)."""
        from src.inference import run_system_d
        sig = inspect.signature(run_system_d)
        assert "weights_path" in sig.parameters

    def test_run_system_d_weights_path_default_is_none(self):
        """weights_path must default to None so existing call sites keep using joint_alpha1.0_weights.pt."""
        from src.inference import run_system_d
        sig = inspect.signature(run_system_d)
        assert sig.parameters["weights_path"].default is None

    def test_run_system_d_uses_weights_path_when_provided(self):
        """run_system_d source must reference weights_path (not just hardcoded joint_alpha1.0_weights.pt)."""
        source = inspect.getsource(__import__("src.inference", fromlist=["run_system_d"]).run_system_d)
        assert "weights_path" in source

    def test_cli_dispatcher_passes_weights_path_to_system_d(self):
        """CLI dispatcher must forward --weights_path to run_system_d (not just system_e)."""
        import src.inference as inf
        cli_source = inspect.getsource(inf)
        # Find the system_d branch and verify weights_path is passed
        system_d_branch = cli_source.split('elif args.system == "system_d":')[1].split("elif args.system ==")[0]
        assert "weights_path=args.weights_path" in system_d_branch


# ---- --thresholds CLI parsing (for System F/G/H single-point runs) ----

class TestThresholdsArgParsing:
    """--thresholds "t0,t1,t2,t3,t4" must parse into a length-5 list of floats."""

    def test_parse_thresholds_importable(self):
        from src.inference import _parse_thresholds_arg  # noqa: F401

    def test_parse_thresholds_returns_list_of_floats(self):
        from src.inference import _parse_thresholds_arg
        out = _parse_thresholds_arg("0.001,0.01,0.3,0.5,0.5")
        assert out == [0.001, 0.01, 0.3, 0.5, 0.5]
        assert all(isinstance(x, float) for x in out)

    def test_parse_thresholds_rejects_wrong_length(self):
        from src.inference import _parse_thresholds_arg
        for bad in ("0.1,0.1,0.1,0.1", "0.1,0.1,0.1,0.1,0.1,0.1", ""):
            with pytest.raises(Exception):
                _parse_thresholds_arg(bad)

    def test_parse_thresholds_rejects_non_float_tokens(self):
        from src.inference import _parse_thresholds_arg
        with pytest.raises(Exception):
            _parse_thresholds_arg("0.1,abc,0.1,0.1,0.1")
