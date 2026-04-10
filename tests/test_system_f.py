"""Tests for System F: per-ramp entropy threshold vector support.

System F = System D α=0.5 weights + per-ramp entropy threshold vector
(length 5, one float per off-ramp) instead of a single scalar threshold.

The underlying inference function is the same compacted early-exit forward
pass used by System C / System D. These tests cover the new vector-input
capability that System F relies on. The existing scalar-input behavior is
covered by tests in tests/test_system_c.py and must remain unchanged.
"""

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
        """Non-float, non-sequence types must raise TypeError."""
        for bad in ("0.1", {"a": 0.1}, None):
            with pytest.raises(TypeError):
                model.forward_compacted_early_exit(
                    small_batch["input_ids"],
                    small_batch["attention_mask"],
                    small_batch["token_type_ids"],
                    entropy_threshold=bad,
                )
