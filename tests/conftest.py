"""Pytest configuration: add project root to sys.path and shared fixtures."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch


@pytest.fixture
def mocked_sweep_env(monkeypatch, tmp_path):
    """Patch GPU-heavy dependencies of run_per_ramp_threshold_sweep so tests
    only exercise partitioning, validation, output-path resolution, and CSV
    writing. Returns the tmp_path that should be passed as ``output_dir``.

    Shared between tests/test_system_f.py and tests/test_system_g.py because
    both files exercise the same shared sweep driver — the only difference is
    which (sweep_subdir, csv_prefix) pair they pass.
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
