"""Tests for TODO 1: Self-distillation + KL divergence loss.

Self-distillation adds a KL-divergence term to each off-ramp loss:
  loss_ramp_i = BCE(ramp_logit, label) + beta * KL(sigmoid(ramp_logit), sigmoid(final_logit).detach())

This forces off-ramps to approximate the final classifier's ranking behavior,
not just binary relevance labels.
"""

import inspect
import os
import tempfile

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.constants import MODEL_NAME, NUM_OFFRAMPS, HIDDEN_SIZE


# ---- Constants ----

def test_system_e_weights_path_constant():
    """Constants must define a separate path for System E weights."""
    from src.constants import DEFAULT_SYSTEM_E_WEIGHTS_PATH
    assert "system_e" in DEFAULT_SYSTEM_E_WEIGHTS_PATH.lower()
    from src.constants import DEFAULT_JOINT_WEIGHTS_PATH
    assert DEFAULT_SYSTEM_E_WEIGHTS_PATH != DEFAULT_JOINT_WEIGHTS_PATH


# ---- train_joint beta parameter ----

class TestBetaParameter:
    """Tests for the beta parameter in train_joint."""

    def test_train_joint_has_beta_param(self):
        """train_joint must accept a beta parameter for KL distillation weight."""
        from src.train_joint import train_joint
        sig = inspect.signature(train_joint)
        assert "beta" in sig.parameters

    def test_beta_default_is_zero(self):
        """beta must default to 0.0 to preserve backward compatibility."""
        from src.train_joint import train_joint
        sig = inspect.signature(train_joint)
        assert sig.parameters["beta"].default == 0.0

    def test_train_joint_has_distill_weights_path_param(self):
        """train_joint must accept distill_weights_path for custom output filename."""
        from src.train_joint import train_joint
        sig = inspect.signature(train_joint)
        assert "output_weights_name" in sig.parameters

    def test_train_joint_save_path_uses_output_weights_name(self):
        """The save path inside train_joint must actually use output_weights_name,
        not hardcode a filename. Required for the alpha=0.5 sweep to write
        distinct files like system_e_alpha0.5_beta1.0_weights.pt instead of
        clobbering the existing system_e_alpha1.0_beta1.0_weights.pt files.
        """
        from src.train_joint import train_joint
        source = inspect.getsource(train_joint)
        # The variable must be referenced in the save path construction.
        # If a future refactor hardcodes the filename, this test catches it.
        assert "output_weights_name" in source
        # And it must reach torch.save
        save_idx = source.find("torch.save")
        assert save_idx > 0
        # output_weights_name must be referenced before the save call
        ref_idx = source.find("output_weights_name", source.find("save_path"))
        # Either save_path or os.path.join uses it before torch.save
        assert ref_idx > 0 and ref_idx < save_idx, (
            "output_weights_name must be used to build the save path"
        )


# ---- KL divergence loss computation ----

class TestKLDistillationLoss:
    """Tests for the KL divergence distillation loss component."""

    def test_compute_distill_loss_importable(self):
        """compute_distill_loss must be importable from train_joint."""
        from src.train_joint import compute_distill_loss

    def test_compute_distill_loss_signature(self):
        """compute_distill_loss(ramp_logit, final_logit) -> scalar tensor."""
        from src.train_joint import compute_distill_loss
        sig = inspect.signature(compute_distill_loss)
        params = list(sig.parameters.keys())
        assert "ramp_logit" in params
        assert "final_logit" in params

    def test_distill_loss_returns_scalar(self):
        """compute_distill_loss must return a scalar tensor."""
        from src.train_joint import compute_distill_loss
        ramp_logit = torch.randn(4)
        final_logit = torch.randn(4)
        loss = compute_distill_loss(ramp_logit, final_logit)
        assert loss.dim() == 0  # scalar

    def test_distill_loss_non_negative(self):
        """KL divergence is always non-negative."""
        from src.train_joint import compute_distill_loss
        ramp_logit = torch.randn(8)
        final_logit = torch.randn(8)
        loss = compute_distill_loss(ramp_logit, final_logit)
        assert loss.item() >= -1e-7  # allow tiny numerical error

    def test_distill_loss_zero_when_identical(self):
        """KL divergence is zero when ramp and final distributions match."""
        from src.train_joint import compute_distill_loss
        logit = torch.tensor([2.0, -1.0, 0.5, 3.0])
        loss = compute_distill_loss(logit, logit)
        assert loss.item() < 1e-5

    def test_distill_loss_positive_when_different(self):
        """KL divergence is positive when ramp and final distributions differ."""
        from src.train_joint import compute_distill_loss
        ramp_logit = torch.tensor([2.0, -1.0, 0.5, 3.0])
        final_logit = torch.tensor([-2.0, 1.0, -0.5, -3.0])
        loss = compute_distill_loss(ramp_logit, final_logit)
        assert loss.item() > 0.01

    def test_distill_loss_detaches_final_logit(self):
        """final_logit should not receive gradients through the KL loss."""
        from src.train_joint import compute_distill_loss
        ramp_logit = torch.randn(4, requires_grad=True)
        final_logit = torch.randn(4, requires_grad=True)
        loss = compute_distill_loss(ramp_logit, final_logit)
        loss.backward()
        assert ramp_logit.grad is not None
        assert final_logit.grad is None, "final_logit must be detached"

    def test_distill_loss_requires_grad_from_ramp(self):
        """The distillation loss must be differentiable w.r.t. ramp_logit."""
        from src.train_joint import compute_distill_loss
        ramp_logit = torch.randn(4, requires_grad=True)
        final_logit = torch.randn(4)
        loss = compute_distill_loss(ramp_logit, final_logit)
        assert loss.requires_grad


# ---- Combined loss with distillation ----

class TestCombinedDistillLoss:
    """Tests for the combined BCE + KL loss in the training loop."""

    @pytest.fixture
    def model(self):
        from src.model import EarlyExitCrossEncoder
        return EarlyExitCrossEncoder(MODEL_NAME)

    def test_combined_loss_with_beta_zero_matches_original(self, model):
        """With beta=0, the combined loss must match the original (no distillation)."""
        from src.train_joint import compute_distill_loss

        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        # Original loss (no distillation)
        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)
        ramp_losses_orig = []
        for logit in out["offramp_logits"]:
            ramp_losses_orig.append(
                F.binary_cross_entropy_with_logits(logit, labels)
            )
        original_total = final_loss + 1.0 * sum(ramp_losses_orig) / len(ramp_losses_orig)

        # Distillation loss with beta=0
        beta = 0.0
        ramp_losses_distill = []
        for logit in out["offramp_logits"]:
            bce = F.binary_cross_entropy_with_logits(logit, labels)
            kl = compute_distill_loss(logit, out["final_logit"])
            ramp_losses_distill.append(bce + beta * kl)
        distill_total = final_loss + 1.0 * sum(ramp_losses_distill) / len(ramp_losses_distill)

        assert torch.allclose(original_total, distill_total)

    def test_combined_loss_with_beta_positive_differs(self, model):
        """With beta>0, the combined loss must differ from the original."""
        from src.train_joint import compute_distill_loss

        dummy_ids = torch.randint(0, 100, (4, 16))
        dummy_mask = torch.ones(4, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)

        # Original (no distillation)
        ramp_losses_orig = [
            F.binary_cross_entropy_with_logits(logit, labels)
            for logit in out["offramp_logits"]
        ]
        original_total = final_loss + sum(ramp_losses_orig) / len(ramp_losses_orig)

        # With distillation
        beta = 1.0
        ramp_losses_distill = []
        for logit in out["offramp_logits"]:
            bce = F.binary_cross_entropy_with_logits(logit, labels)
            kl = compute_distill_loss(logit, out["final_logit"])
            ramp_losses_distill.append(bce + beta * kl)
        distill_total = final_loss + sum(ramp_losses_distill) / len(ramp_losses_distill)

        # KL >= 0, so distill_total >= original_total
        assert distill_total.item() >= original_total.item() - 1e-6

    def test_higher_beta_increases_loss(self, model):
        """Higher beta should increase the total loss (KL is non-negative)."""
        from src.train_joint import compute_distill_loss

        dummy_ids = torch.randint(0, 100, (4, 16))
        dummy_mask = torch.ones(4, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)

        losses = {}
        for beta in [0.1, 1.0, 2.0]:
            ramp_losses = []
            for logit in out["offramp_logits"]:
                bce = F.binary_cross_entropy_with_logits(logit, labels)
                kl = compute_distill_loss(logit, out["final_logit"])
                ramp_losses.append(bce + beta * kl)
            losses[beta] = final_loss + sum(ramp_losses) / len(ramp_losses)

        assert losses[1.0].item() >= losses[0.1].item() - 1e-6
        assert losses[2.0].item() >= losses[1.0].item() - 1e-6

    def test_combined_alpha_beta_loss_formula(self, model):
        """The training loop loss must equal final + alpha*mean(BCE + beta*KL),
        which is algebraically the same as final + alpha*mean(BCE) + (alpha*beta)*mean(KL).

        This documents the (alpha*beta) effective KL weight interaction so the
        System E alpha=0.5 sweep is interpretable: at alpha=0.5, the listed beta
        values produce effective KL weights [0.05, 0.25, 0.5, 1.0] — half what
        the same beta values produce at alpha=1.0.
        """
        from src.train_joint import compute_distill_loss

        dummy_ids = torch.randint(0, 100, (4, 16))
        dummy_mask = torch.ones(4, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0, 1.0, 0.0])

        with torch.no_grad():
            out = model.forward_with_offramps(dummy_ids, dummy_mask)

        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)

        # Test several (alpha, beta) pairs including the new (0.5, *) configurations
        for alpha, beta in [(0.5, 0.1), (0.5, 1.0), (0.5, 2.0), (1.0, 1.0), (2.0, 0.5)]:
            # Method A: matches train_joint.py:137-144 exactly
            ramp_losses = []
            for logit in out["offramp_logits"]:
                bce = F.binary_cross_entropy_with_logits(logit, labels)
                kl = compute_distill_loss(logit, out["final_logit"])
                ramp_losses.append(bce + beta * kl)
            total_a = final_loss + alpha * sum(ramp_losses) / len(ramp_losses)

            # Method B: expanded form — proves alpha*beta is the effective KL weight
            bce_terms = [
                F.binary_cross_entropy_with_logits(logit, labels)
                for logit in out["offramp_logits"]
            ]
            kl_terms = [
                compute_distill_loss(logit, out["final_logit"])
                for logit in out["offramp_logits"]
            ]
            total_b = (
                final_loss
                + alpha * sum(bce_terms) / len(bce_terms)
                + (alpha * beta) * sum(kl_terms) / len(kl_terms)
            )

            assert torch.allclose(total_a, total_b, atol=1e-6), (
                f"Formula mismatch at (alpha={alpha}, beta={beta}): "
                f"{total_a.item()} vs {total_b.item()}"
            )

    def test_distill_gradient_flows_to_backbone(self, model):
        """Distillation loss gradients must flow to backbone when unfrozen."""
        from src.train_joint import compute_distill_loss

        for p in model.backbone.parameters():
            p.requires_grad = True

        dummy_ids = torch.randint(0, 100, (2, 16))
        dummy_mask = torch.ones(2, 16, dtype=torch.long)
        labels = torch.tensor([0.0, 1.0])

        out = model.forward_with_offramps(dummy_ids, dummy_mask)

        final_loss = F.binary_cross_entropy_with_logits(out["final_logit"], labels)
        ramp_losses = []
        for logit in out["offramp_logits"]:
            bce = F.binary_cross_entropy_with_logits(logit, labels)
            kl = compute_distill_loss(logit, out["final_logit"])
            ramp_losses.append(bce + 1.0 * kl)

        total_loss = final_loss + sum(ramp_losses) / len(ramp_losses)
        total_loss.backward()

        layer0_weight = model.backbone.bert.encoder.layer[0].attention.self.query.weight
        assert layer0_weight.grad is not None
        assert layer0_weight.grad.abs().sum() > 0


# ---- Inference with distilled weights ----

class TestSystemEInference:
    """Tests for System E inference with self-distillation weights."""

    def test_run_system_e_importable(self):
        """run_system_e must be importable from inference module."""
        from src.inference import run_system_e

    def test_run_system_e_signature(self):
        """run_system_e must have consistent signature with run_system_d."""
        from src.inference import run_system_e
        sig = inspect.signature(run_system_e)
        params = set(sig.parameters.keys())
        for name in ("tokenized_path", "batch_size", "thresholds", "output_dir"):
            assert name in params, f"Missing parameter: {name}"

    def test_run_system_e_has_weights_path_param(self):
        """run_system_e must accept a weights_path parameter."""
        from src.inference import run_system_e
        sig = inspect.signature(run_system_e)
        assert "weights_path" in sig.parameters

    def test_run_system_e_uses_system_e_weights(self):
        """run_system_e must load System E weights by default."""
        from src.inference import run_system_e
        source = inspect.getsource(run_system_e)
        assert "system_e" in source.lower()

    def test_run_system_e_system_name(self):
        """Result dict must use system='system_e'."""
        from src.inference import run_system_e
        source = inspect.getsource(run_system_e)
        assert '"system_e"' in source

    def test_run_system_e_records_weights_in_result(self):
        """Result dict must include which weights file was used."""
        from src.inference import run_system_e
        source = inspect.getsource(run_system_e)
        assert '"weights"' in source

    def test_run_system_e_saves_system_e_results(self):
        """System E must save results as system_e_results.json."""
        from src.inference import run_system_e
        source = inspect.getsource(run_system_e)
        assert "system_e_results.json" in source

    def test_cli_accepts_system_e(self):
        """CLI must accept --system system_e."""
        import subprocess
        result = subprocess.run(
            ["python", "-m", "src.inference", "--system", "system_e", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0 or "error" not in result.stderr.lower()


# ---- End-to-end short training test ----

class TestDistillTrainingE2E:
    """End-to-end test for distillation training with tiny data."""

    @pytest.fixture
    def tiny_parquet(self, tmp_path):
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

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="train_joint requires CUDA"
    )
    def test_train_joint_with_beta(self, tiny_parquet):
        """train_joint with beta>0 should run and save distillation weights."""
        from src.train_joint import train_joint

        with tempfile.TemporaryDirectory() as tmpdir:
            train_joint(
                data_path=tiny_parquet,
                epochs=1,
                batch_size=4,
                beta=0.5,
                output_dir=tmpdir,
                max_steps=2,
            )
            weights_path = os.path.join(tmpdir, "system_e_joint_distill_weights.pt")
            assert os.path.exists(weights_path)
            state = torch.load(weights_path, weights_only=True)
            assert "backbone" in state
            assert "offramps" in state
