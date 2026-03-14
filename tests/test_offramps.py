"""Tests for off-ramp architecture (src/offramps.py and src/model.py)."""

import math

import torch
import pytest


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    from src.model import EarlyExitCrossEncoder
    return EarlyExitCrossEncoder()


class TestOffRampHead:
    def test_output_shape(self):
        from src.offramps import OffRampHead
        ramp = OffRampHead(hidden_size=384)
        x = torch.randn(4, 128, 384)
        out = ramp(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_entropy_shape(self):
        from src.offramps import OffRampHead
        ramp = OffRampHead(hidden_size=384)
        logit = torch.randn(4)
        entropy = ramp.compute_entropy(logit)
        assert entropy.shape == (4,)

    def test_entropy_range(self):
        from src.offramps import OffRampHead
        ramp = OffRampHead(hidden_size=384)
        logit = torch.randn(100)
        entropy = ramp.compute_entropy(logit)
        assert (entropy >= 0).all(), "Entropy should be non-negative"
        assert (entropy <= math.log(2) + 1e-6).all(), "Entropy should be <= ln(2)"

    def test_entropy_max_at_zero_logit(self):
        """Entropy is maximized when sigmoid(logit) = 0.5, i.e., logit = 0."""
        from src.offramps import OffRampHead
        ramp = OffRampHead(hidden_size=384)
        logit = torch.tensor([0.0])
        entropy = ramp.compute_entropy(logit)
        assert abs(entropy.item() - math.log(2)) < 1e-5

    def test_entropy_low_for_confident(self):
        """Large absolute logit → low entropy."""
        from src.offramps import OffRampHead
        ramp = OffRampHead(hidden_size=384)
        logit = torch.tensor([10.0, -10.0])
        entropy = ramp.compute_entropy(logit)
        assert (entropy < 0.01).all()


class TestOffRampCollection:
    def test_num_ramps(self):
        from src.offramps import OffRampCollection
        collection = OffRampCollection(num_ramps=5, hidden_size=384)
        assert len(collection.ramps) == 5

    def test_routing(self):
        from src.offramps import OffRampCollection
        collection = OffRampCollection(num_ramps=5, hidden_size=384)
        x = torch.randn(2, 64, 384)
        for i in range(5):
            out = collection(i, x)
            assert out.shape == (2,)


class TestEarlyExitCrossEncoder:
    def test_backbone_frozen(self, model):
        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone params should be frozen"

    def test_offramps_trainable(self, model):
        for p in model.offramps.parameters():
            assert p.requires_grad, "Off-ramp params should be trainable"

    def test_forward_with_offramps(self, model):
        tokenizer = model.tokenizer
        encoded = tokenizer(
            ["what is python"],
            ["Python is a programming language"],
            max_length=32,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model.forward_with_offramps(
                encoded["input_ids"],
                encoded["attention_mask"],
                encoded["token_type_ids"],
            )
        assert "final_logit" in out
        assert "offramp_logits" in out
        assert "offramp_entropies" in out
        assert len(out["offramp_logits"]) == 5
        assert len(out["offramp_entropies"]) == 5
        assert out["final_logit"].shape == (1,)
        for logit in out["offramp_logits"]:
            assert logit.shape == (1,)
        for entropy in out["offramp_entropies"]:
            assert entropy.shape == (1,)

    def test_offramp_param_count(self, model):
        """5 ramps × (384 weights + 1 bias) = 1925 trainable params."""
        trainable = sum(p.numel() for p in model.offramps.parameters())
        assert trainable == 5 * (384 + 1), f"Expected 1925, got {trainable}"
