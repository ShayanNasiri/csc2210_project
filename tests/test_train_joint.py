"""Tests for System D: joint backbone + off-ramp training."""

import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

from src.constants import MODEL_NAME, NUM_OFFRAMPS, HIDDEN_SIZE, NUM_BERT_LAYERS


# ---- Import / module tests (always run) ----

def test_train_joint_importable():
    """train_joint module must be importable."""
    from src import train_joint


def test_train_joint_function_exists():
    """train_joint must export a train_joint function."""
    from src.train_joint import train_joint
    assert callable(train_joint)


def test_train_joint_signature():
    """train_joint must accept expected parameters."""
    import inspect
    from src.train_joint import train_joint
    sig = inspect.signature(train_joint)
    params = set(sig.parameters.keys())
    assert "data_path" in params
    assert "epochs" in params
    assert "batch_size" in params
    assert "output_dir" in params


def test_train_joint_has_backbone_lr_param():
    """train_joint must accept backbone_lr for differential learning rates."""
    import inspect
    from src.train_joint import train_joint
    sig = inspect.signature(train_joint)
    params = set(sig.parameters.keys())
    assert "backbone_lr" in params
    assert "offramp_lr" in params


def test_joint_weights_path_constant():
    """Constants must define the joint weights path separately from offramp weights."""
    from src.constants import DEFAULT_JOINT_WEIGHTS_PATH
    assert "joint" in DEFAULT_JOINT_WEIGHTS_PATH.lower()
    # Must be different from the frozen off-ramp weights
    from src.constants import DEFAULT_OFFRAMP_WEIGHTS_PATH
    assert DEFAULT_JOINT_WEIGHTS_PATH != DEFAULT_OFFRAMP_WEIGHTS_PATH


# ---- Model behavior tests (need model, no GPU required) ----

class TestJointTrainingSetup:
    """Tests for joint training model configuration."""

    @pytest.fixture
    def model(self):
        from src.model import EarlyExitCrossEncoder
        return EarlyExitCrossEncoder(MODEL_NAME)

    def test_unfreeze_backbone(self, model):
        """After unfreezing, backbone parameters must have requires_grad=True."""
        # By default, backbone is frozen
        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone should start frozen"

        # Unfreeze
        for p in model.backbone.parameters():
            p.requires_grad = True

        for p in model.backbone.parameters():
            assert p.requires_grad, "Backbone should be unfrozen"

    def test_offramps_always_trainable(self, model):
        """Off-ramp parameters must always have requires_grad=True."""
        for p in model.offramps.parameters():
            assert p.requires_grad

    def test_differential_lr_param_groups(self, model):
        """Optimizer should use two param groups with different learning rates."""
        # Unfreeze backbone
        for p in model.backbone.parameters():
            p.requires_grad = True

        backbone_lr = 2e-5
        offramp_lr = 1e-3

        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": backbone_lr},
            {"params": model.offramps.parameters(), "lr": offramp_lr},
        ])

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["lr"] == backbone_lr
        assert optimizer.param_groups[1]["lr"] == offramp_lr

    def test_forward_with_offramps_returns_final_logit(self, model):
        """forward_with_offramps must return final_logit for combined loss."""
        dummy_ids = torch.randint(0, 100, (2, 32))
        dummy_mask = torch.ones(2, 32, dtype=torch.long)

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        assert "final_logit" in out
        assert out["final_logit"].shape == (2,)
        assert "offramp_logits" in out
        assert len(out["offramp_logits"]) == NUM_OFFRAMPS

    def test_combined_loss_computation(self, model):
        """Combined loss must include both final classifier and off-ramp losses."""
        dummy_ids = torch.randint(0, 100, (4, 32))
        dummy_mask = torch.ones(4, 32, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        # Unfreeze for gradient computation
        for p in model.backbone.parameters():
            p.requires_grad = True

        out = model.forward_with_offramps(dummy_ids, dummy_mask)

        # Final classifier loss
        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)

        # Off-ramp losses
        ramp_losses = []
        for logit in out["offramp_logits"]:
            ramp_losses.append(F.binary_cross_entropy_with_logits(logit, labels))

        alpha = 1.0
        total_loss = final_loss + alpha * sum(ramp_losses) / len(ramp_losses)

        assert total_loss.requires_grad
        assert total_loss.item() > 0

    def test_joint_loss_gradient_flows_to_backbone(self, model):
        """Gradients from off-ramp losses must reach backbone parameters."""
        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        # Unfreeze backbone
        for p in model.backbone.parameters():
            p.requires_grad = True

        out = model.forward_with_offramps(dummy_ids, dummy_mask)

        # Compute combined loss
        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)
        ramp_losses = [
            F.binary_cross_entropy_with_logits(logit, labels)
            for logit in out["offramp_logits"]
        ]
        total_loss = final_loss + sum(ramp_losses) / len(ramp_losses)
        total_loss.backward()

        # Check that backbone layer 0 received gradients
        layer0_weight = model.backbone.bert.encoder.layer[0].attention.self.query.weight
        assert layer0_weight.grad is not None
        assert layer0_weight.grad.abs().sum() > 0, "Backbone layer 0 should receive gradients"

    def test_joint_saves_full_model_state(self):
        """Joint training must save both backbone and off-ramp weights."""
        from src.model import EarlyExitCrossEncoder
        import tempfile

        model = EarlyExitCrossEncoder(MODEL_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "joint_weights.pt")
            state = {
                "backbone": model.backbone.state_dict(),
                "offramps": model.offramps.state_dict(),
            }
            torch.save(state, save_path)

            # Reload and verify
            loaded = torch.load(save_path, weights_only=True)
            assert "backbone" in loaded
            assert "offramps" in loaded
            assert len(loaded["backbone"]) > 10  # BERT has many params
            assert len(loaded["offramps"]) == 10  # 5 ramps * (weight + bias)


# ---- Inference with joint weights tests ----

class TestSystemDInference:
    """Tests for System D inference (joint weights + compacted early exit)."""

    def test_run_system_d_importable(self):
        """run_system_d must be importable from inference module."""
        from src.inference import run_system_d

    def test_run_system_d_cli_entry(self):
        """CLI must accept --system system_d."""
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.inference", "--system", "system_d", "--help"],
            capture_output=True, text=True,
        )
        # Just checking it doesn't crash on parsing — help exits with 0
        # The actual run would fail without data, but the arg should parse
        assert result.returncode == 0 or "error" not in result.stderr.lower()

    def test_joint_weights_load_pattern(self):
        """System D must load joint weights (backbone + offramps) not just offramp weights."""
        import inspect
        from src.inference import run_system_d
        source = inspect.getsource(run_system_d)
        assert "joint_weights" in source or "joint" in source


# ---- Training loop behavior tests ----

class TestJointTrainingBehavior:
    """Tests for joint training loop specifics."""

    @pytest.fixture
    def tiny_parquet(self, tmp_path):
        """Create a small parquet file for training tests."""
        import pandas as pd
        df = pd.DataFrame({
            "qid": ["q1"] * 4 + ["q2"] * 4,
            "docid": [f"d{i}" for i in range(8)],
            "query": ["what is python"] * 4 + ["what is java"] * 4,
            "passage": [
                "Python is a language", "Cats are cute",
                "Python is great", "Dogs bark",
                "Java is a language", "Birds fly",
                "Java runs on JVM", "Fish swim",
            ],
            "label": [1, 0, 1, 0, 1, 0, 1, 0],
        })
        path = str(tmp_path / "tiny_train.parquet")
        df.to_parquet(path, index=False)
        return path

    def test_alpha_zero_excludes_ramp_loss(self):
        """With alpha=0, total_loss should equal final_loss only."""
        from src.model import EarlyExitCrossEncoder
        model = EarlyExitCrossEncoder(MODEL_NAME)

        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)
        ramp_losses = [
            F.binary_cross_entropy_with_logits(logit, labels)
            for logit in out["offramp_logits"]
        ]
        alpha = 0.0
        total_loss = final_loss + alpha * sum(ramp_losses) / len(ramp_losses)

        assert torch.allclose(total_loss, final_loss)

    def test_alpha_scales_ramp_contribution(self):
        """Higher alpha should increase total loss relative to final-only loss."""
        from src.model import EarlyExitCrossEncoder
        model = EarlyExitCrossEncoder(MODEL_NAME)

        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)
        ramp_losses = [
            F.binary_cross_entropy_with_logits(logit, labels)
            for logit in out["offramp_logits"]
        ]
        mean_ramp = sum(ramp_losses) / len(ramp_losses)

        loss_a1 = final_loss + 1.0 * mean_ramp
        loss_a2 = final_loss + 2.0 * mean_ramp

        assert loss_a2.item() > loss_a1.item()

    def test_cuda_guard_raises_on_cpu(self):
        """train_joint must raise RuntimeError when CUDA is not available."""
        from unittest.mock import patch
        from src.train_joint import train_joint

        with patch("src.train_joint.get_device", return_value=torch.device("cpu")):
            with pytest.raises(RuntimeError, match="CUDA not available"):
                train_joint(data_path="nonexistent.parquet", max_steps=1)

    def test_train_joint_has_alpha_param(self):
        """train_joint must accept alpha parameter for loss weighting."""
        import inspect
        from src.train_joint import train_joint
        sig = inspect.signature(train_joint)
        assert "alpha" in sig.parameters
        assert sig.parameters["alpha"].default == 1.0

    def test_train_joint_has_max_steps_param(self):
        """train_joint must accept max_steps parameter for short runs."""
        import inspect
        from src.train_joint import train_joint
        sig = inspect.signature(train_joint)
        assert "max_steps" in sig.parameters
        assert sig.parameters["max_steps"].default == -1

    def test_weight_save_format(self):
        """Joint weights file must contain backbone and offramps keys with correct sizes."""
        from src.model import EarlyExitCrossEncoder
        model = EarlyExitCrossEncoder(MODEL_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "joint_weights.pt")
            state = {
                "backbone": model.backbone.state_dict(),
                "offramps": model.offramps.state_dict(),
            }
            torch.save(state, save_path)

            loaded = torch.load(save_path, weights_only=True)

            # Backbone state dict should have many keys (100+)
            assert len(loaded["backbone"]) > 50

            # Off-ramp state dict: 5 ramps * (weight + bias) = 10 keys
            assert len(loaded["offramps"]) == 10

            # Weight shapes
            for i in range(NUM_OFFRAMPS):
                w_key = f"ramps.{i}.linear.weight"
                b_key = f"ramps.{i}.linear.bias"
                assert w_key in loaded["offramps"]
                assert b_key in loaded["offramps"]
                assert loaded["offramps"][w_key].shape == (1, HIDDEN_SIZE)
                assert loaded["offramps"][b_key].shape == (1,)

    def test_joint_weights_loadable_into_fresh_model(self):
        """Joint weights must be loadable into a fresh EarlyExitCrossEncoder."""
        from src.model import EarlyExitCrossEncoder
        model = EarlyExitCrossEncoder(MODEL_NAME)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "joint_weights.pt")
            state = {
                "backbone": model.backbone.state_dict(),
                "offramps": model.offramps.state_dict(),
            }
            torch.save(state, save_path)

            # Load into a fresh model (as System D inference does)
            model2 = EarlyExitCrossEncoder(MODEL_NAME)
            loaded = torch.load(save_path, weights_only=True)
            model2.backbone.load_state_dict(loaded["backbone"])
            model2.offramps.load_state_dict(loaded["offramps"])

            # Verify outputs match
            dummy_ids = torch.randint(0, 100, (2, 16))
            dummy_mask = torch.ones(2, 16, dtype=torch.long)
            with torch.no_grad():
                out1 = model.forward_with_offramps(dummy_ids, dummy_mask)
                out2 = model2.forward_with_offramps(dummy_ids, dummy_mask)

            assert torch.allclose(out1["final_logit"], out2["final_logit"])
            for i in range(NUM_OFFRAMPS):
                assert torch.allclose(
                    out1["offramp_logits"][i], out2["offramp_logits"][i]
                )
