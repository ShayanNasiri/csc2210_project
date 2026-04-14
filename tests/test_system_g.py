"""Tests for System G: per-ramp entropy threshold grid sweep on System E
(α=1.0, β=1.0) weights.

System G reuses the same shared sweep driver ``run_per_ramp_threshold_sweep``
that System F uses — the difference is purely in the kwargs passed by
``scripts/run_system_g_per_ramp_threshold_sweep.sh``:

* ``--weights_path results/system_e_beta1.0_weights.pt``
* ``--sweep_subdir system_g_sweep_results``
* ``--csv_prefix system_g_e_alpha1.0_beta1.0``

The driver-contract tests for the new ``sweep_subdir`` / ``csv_prefix``
kwargs themselves live in ``tests/test_system_f.py``. This file owns the
end-to-end "System G layout" assertion: every (t0, t1) cell of the full
7^5 grid lands at the System G output path with the verbose System G prefix,
AND the System F output directory is left untouched (collision safety).

The shared ``mocked_sweep_env`` fixture is defined in ``tests/conftest.py``.
"""

import os

import pytest


class TestSystemGSweepLayout:
    """End-to-end output-layout assertions for the System G sweep."""

    def test_system_g_full_grid_layout(self, mocked_sweep_env):
        """All 49 (t0, t1) cells must land at
        ``<output_dir>/system_g_sweep_results/system_g_e_alpha1.0_beta1.0_t0=*_t1=*.csv``,
        and the System F default subdir must NOT be created.
        """
        from src.constants import DEFAULT_PER_RAMP_GRID
        from src.inference import run_per_ramp_threshold_sweep

        produced_paths = []
        for task_id in range(49):
            csv_path = run_per_ramp_threshold_sweep(
                task_id=task_id,
                num_tasks=49,
                grid_values=DEFAULT_PER_RAMP_GRID,
                output_dir=str(mocked_sweep_env),
                sweep_subdir="system_g_sweep_results",
                csv_prefix="system_g_e_alpha1.0_beta1.0",
            )
            produced_paths.append(csv_path)

        expected_basenames = {
            f"system_g_e_alpha1.0_beta1.0_t0={t0}_t1={t1}.csv"
            for t0 in DEFAULT_PER_RAMP_GRID
            for t1 in DEFAULT_PER_RAMP_GRID
        }
        produced_basenames = {os.path.basename(p) for p in produced_paths}
        assert produced_basenames == expected_basenames, (
            f"System G grid coverage broken. "
            f"Missing: {expected_basenames - produced_basenames}. "
            f"Extra: {produced_basenames - expected_basenames}."
        )
        assert len(produced_paths) == 49
        assert len(set(produced_paths)) == 49, (
            "Some (t0, t1) cells were produced twice"
        )

        expected_dir = os.path.join(str(mocked_sweep_env), "system_g_sweep_results")
        for p in produced_paths:
            assert os.path.dirname(p) == expected_dir, (
                f"System G run leaked outside its subdir: {p}"
            )
            assert os.path.isfile(p), f"CSV not actually written: {p}"

        # Collision safety: a System G run must not create the System F dir.
        # Re-running System G against an output_dir that also holds committed
        # System F results must NEVER touch them.
        assert not os.path.isdir(
            os.path.join(str(mocked_sweep_env), "system_f_sweep_results")
        ), "System G run leaked into the System F subdir — collision risk"


# ---------------------------------------------------------------------------
# run_system_g — single-operating-point runner tests
# ---------------------------------------------------------------------------

class TestRunSystemG:
    def test_importable(self):
        from src.inference import run_system_g  # noqa: F401

    def test_signature(self):
        import inspect
        from src.inference import run_system_g
        sig = inspect.signature(run_system_g)
        for name in ("tokenized_path", "batch_size", "thresholds",
                     "output_dir", "weights_path", "results_tag"):
            assert name in sig.parameters, f"missing param {name}"

    def test_wrong_length_thresholds_raises(self, mocked_sweep_env):
        from src.inference import run_system_g
        with pytest.raises(ValueError, match="length 5"):
            run_system_g(
                thresholds=[0.1, 0.1, 0.1],
                output_dir=str(mocked_sweep_env),
                weights_path="irrelevant.pt",
            )

    def test_writes_json_with_system_g_label(self, mocked_sweep_env):
        import json
        from src.inference import run_system_g
        result = run_system_g(
            thresholds=[0.001, 0.01, 0.1, 0.5, 0.003],
            output_dir=str(mocked_sweep_env),
            weights_path="irrelevant.pt",
            results_tag="test_final_",
        )
        assert result["system"] == "system_g"
        assert result["patience"] == 1
        json_path = os.path.join(str(mocked_sweep_env), "test_final_system_g_results.json")
        assert os.path.isfile(json_path)
        with open(json_path) as f:
            on_disk = json.load(f)
        assert on_disk[0]["system"] == "system_g"

    def test_cli_dispatch_routes_to_run_system_g(self):
        import inspect
        import src.inference as inf
        src = inspect.getsource(inf)
        branch = src.split('elif args.system == "system_g":')[1].split("elif args.system ==")[0]
        assert "run_system_g(" in branch
        assert "thresholds=" in branch
        assert "weights_path=args.weights_path" in branch
