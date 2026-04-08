"""Tests for src/train_offramps.py — off-ramp training with frozen backbone."""

import inspect
import os
import tempfile

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.constants import MODEL_NAME, NUM_OFFRAMPS, HIDDEN_SIZE, MAX_TOKEN_LENGTH


# ---- Import / module tests ----

def test_train_offramps_importable():
    from src import train_offramps


def test_train_offramps_function_exists():
    from src.train_offramps import train_offramps
    assert callable(train_offramps)


def test_tokenize_training_data_function_exists():
    from src.train_offramps import tokenize_training_data
    assert callable(tokenize_training_data)


def test_train_offramps_signature():
    from src.train_offramps import train_offramps
    sig = inspect.signature(train_offramps)
    params = set(sig.parameters.keys())
    for name in ("data_path", "epochs", "batch_size", "lr", "output_dir", "max_steps"):
        assert name in params, f"Missing parameter: {name}"


def test_tokenize_training_data_signature():
    from src.train_offramps import tokenize_training_data
    sig = inspect.signature(tokenize_training_data)
    params = set(sig.parameters.keys())
    assert "data_path" in params
    assert "batch_size" in params


# ---- Tokenization tests ----

@pytest.fixture
def tiny_parquet(tmp_path):
    """Create a small parquet file with 8 query-passage pairs."""
    df = pd.DataFrame({
        "qid": ["q1"] * 4 + ["q2"] * 4,
        "docid": [f"d{i}" for i in range(8)],
        "query": ["what is python"] * 4 + ["what is java"] * 4,
        "passage": [
            "Python is a language", "Cats are cute", "Python is great", "Dogs bark",
            "Java is a language", "Birds fly", "Java runs on JVM", "Fish swim",
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0],
    })
    path = str(tmp_path / "tiny_train.parquet")
    df.to_parquet(path, index=False)
    return path


def test_tokenize_returns_four_tensors(tiny_parquet):
    from src.train_offramps import tokenize_training_data
    input_ids, attention_mask, token_type_ids, labels = tokenize_training_data(
        tiny_parquet, batch_size=4
    )
    assert isinstance(input_ids, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert isinstance(token_type_ids, torch.Tensor)
    assert isinstance(labels, torch.Tensor)


def test_tokenize_shapes(tiny_parquet):
    from src.train_offramps import tokenize_training_data
    input_ids, attention_mask, token_type_ids, labels = tokenize_training_data(
        tiny_parquet, batch_size=4
    )
    assert input_ids.shape[0] == 8
    assert input_ids.shape[1] == MAX_TOKEN_LENGTH
    assert attention_mask.shape == input_ids.shape
    assert token_type_ids.shape == input_ids.shape
    assert labels.shape == (8,)


def test_tokenize_labels_match_source(tiny_parquet):
    from src.train_offramps import tokenize_training_data
    _, _, _, labels = tokenize_training_data(tiny_parquet, batch_size=100)
    assert labels.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]


def test_tokenize_label_dtype(tiny_parquet):
    from src.train_offramps import tokenize_training_data
    _, _, _, labels = tokenize_training_data(tiny_parquet)
    assert labels.dtype == torch.float32


# ---- Training loop tests ----

class TestTrainOfframpsLoop:
    """Tests for the train_offramps training loop logic."""

    @pytest.fixture
    def model(self):
        from src.model import EarlyExitCrossEncoder
        return EarlyExitCrossEncoder(MODEL_NAME)

    def test_backbone_stays_frozen(self, model):
        """During off-ramp training, backbone must stay frozen."""
        for p in model.backbone.parameters():
            assert not p.requires_grad

    def test_only_offramp_params_optimized(self, model):
        """Optimizer should only contain off-ramp parameters."""
        optimizer = torch.optim.AdamW(model.offramps.parameters(), lr=1e-3)
        total_optimized = sum(
            p.numel() for group in optimizer.param_groups for p in group["params"]
        )
        total_offramp = sum(p.numel() for p in model.offramps.parameters())
        assert total_optimized == total_offramp

    def test_offramp_loss_is_bce(self, model):
        """Off-ramp training uses BCE loss on each ramp."""
        dummy_ids = torch.randint(0, 100, (2, 32))
        dummy_mask = torch.ones(2, 32, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        for logit in out["offramp_logits"]:
            loss = F.binary_cross_entropy_with_logits(logit, labels)
            assert loss.item() > 0

    def test_offramp_loss_is_mean_of_ramps(self, model):
        """Total loss is mean of per-ramp BCE losses."""
        dummy_ids = torch.randint(0, 100, (2, 32))
        dummy_mask = torch.ones(2, 32, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        out = model.forward_with_offramps(dummy_ids, dummy_mask)

        losses = []
        for logit in out["offramp_logits"]:
            losses.append(F.binary_cross_entropy_with_logits(logit, labels))

        total_loss = sum(losses) / len(losses)
        assert total_loss.requires_grad
        assert len(losses) == NUM_OFFRAMPS

    def test_gradient_does_not_reach_backbone(self, model):
        """Backbone gradients must be None since backbone is frozen."""
        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        out = model.forward_with_offramps(dummy_ids, dummy_mask)
        losses = [
            F.binary_cross_entropy_with_logits(logit, labels)
            for logit in out["offramp_logits"]
        ]
        total_loss = sum(losses) / len(losses)
        total_loss.backward()

        layer0_weight = model.backbone.bert.encoder.layer[0].attention.self.query.weight
        assert layer0_weight.grad is None, "Backbone should not receive gradients"

    def test_gradient_reaches_offramps(self, model):
        """Off-ramp parameters must receive gradients."""
        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        out = model.forward_with_offramps(dummy_ids, dummy_mask)
        losses = [
            F.binary_cross_entropy_with_logits(logit, labels)
            for logit in out["offramp_logits"]
        ]
        total_loss = sum(losses) / len(losses)
        total_loss.backward()

        for ramp in model.offramps.ramps:
            assert ramp.linear.weight.grad is not None
            assert ramp.linear.weight.grad.abs().sum() > 0


# ---- Weight saving tests ----

class TestOfframpWeightSaving:
    """Tests for off-ramp weight saving format."""

    def test_saves_state_dict_keys(self):
        from src.model import EarlyExitCrossEncoder
        model = EarlyExitCrossEncoder(MODEL_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "offramp_weights.pt")
            torch.save(model.offramps.state_dict(), save_path)

            loaded = torch.load(save_path, weights_only=True)
            assert len(loaded) == 10  # 5 ramps * (weight + bias)

    def test_roundtrip_preserves_values(self):
        from src.model import EarlyExitCrossEncoder
        model = EarlyExitCrossEncoder(MODEL_NAME)

        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)

        with torch.no_grad():
            out_before = model.forward_with_offramps(dummy_ids, dummy_mask)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "offramp_weights.pt")
            torch.save(model.offramps.state_dict(), save_path)

            # Load into a fresh model
            model2 = EarlyExitCrossEncoder(MODEL_NAME)
            model2.offramps.load_state_dict(
                torch.load(save_path, weights_only=True)
            )

            with torch.no_grad():
                out_after = model2.forward_with_offramps(dummy_ids, dummy_mask)

        for i in range(NUM_OFFRAMPS):
            assert torch.allclose(
                out_before["offramp_logits"][i],
                out_after["offramp_logits"][i],
            )


# ---- End-to-end training test (short run) ----

def test_train_offramps_max_steps(tiny_parquet):
    """train_offramps with max_steps=2 should run and save weights."""
    from src.train_offramps import train_offramps

    with tempfile.TemporaryDirectory() as tmpdir:
        train_offramps(
            data_path=tiny_parquet,
            epochs=1,
            batch_size=4,
            lr=1e-3,
            output_dir=tmpdir,
            max_steps=2,
        )
        weights_path = os.path.join(tmpdir, "offramp_weights.pt")
        assert os.path.exists(weights_path)
        state = torch.load(weights_path, weights_only=True)
        assert len(state) == 10  # 5 ramps * (weight + bias)
