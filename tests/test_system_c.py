"""Tests for System C: Triton-compacted early-exit inference (Phase 6).

Unit tests (TestForwardCompactedEarlyExit) do not require CUDA or data files.
Inference tests (TestRunSystemC) require CUDA + data/dev_tokenized.pt.
"""

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
# Unit tests: forward_compacted_early_exit
# ---------------------------------------------------------------------------

class TestForwardCompactedEarlyExit:

    def test_method_exists(self, model):
        assert hasattr(model, "forward_compacted_early_exit"), (
            "EarlyExitCrossEncoder must have forward_compacted_early_exit method"
        )

    def test_return_keys(self, model, small_batch):
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
        assert "scores" in out, "Return dict must contain 'scores'"
        assert "exit_layer" in out, "Return dict must contain 'exit_layer'"
        assert "exit_counts" in out, "Return dict must contain 'exit_counts'"

    def test_scores_shape(self, model, small_batch):
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
        assert out["scores"].shape == (4,), (
            f"Expected scores shape (4,), got {out['scores'].shape}"
        )

    def test_exit_layer_shape_and_range(self, model, small_batch):
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
        el = out["exit_layer"]
        assert el.shape == (4,), f"Expected exit_layer shape (4,), got {el.shape}"
        assert el.min().item() >= 0, "exit_layer values must be >= 0"
        assert el.max().item() <= 5, "exit_layer values must be <= 5 (5 = final layer)"

    def test_exit_counts_length(self, model, small_batch):
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
        assert len(out["exit_counts"]) == 6, (
            f"exit_counts must have 6 entries (5 off-ramps + final), got {len(out['exit_counts'])}"
        )

    def test_exit_counts_sum_equals_batch(self, model, small_batch):
        batch_size = small_batch["input_ids"].shape[0]
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
        total = sum(out["exit_counts"])
        assert total == batch_size, (
            f"exit_counts must sum to batch_size={batch_size}, got {total}"
        )

    def test_threshold_zero_nothing_exits(self, model, small_batch):
        """With threshold=0.0, entropy is never < 0 so no doc exits early."""
        batch_size = small_batch["input_ids"].shape[0]
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.0,
            )
        counts = out["exit_counts"]
        assert counts[5] == batch_size, (
            f"With threshold=0, all docs must reach final layer. "
            f"exit_counts={counts}"
        )
        assert sum(counts[:5]) == 0, (
            f"With threshold=0, no docs should exit at off-ramps. "
            f"exit_counts={counts}"
        )

    def test_threshold_one_all_exit_at_first_ramp(self, model, small_batch):
        """With threshold=1.0, every doc exits at the first off-ramp.

        Max binary entropy is ln(2) ≈ 0.693 < 1.0, so all entropies satisfy
        entropy < 1.0, triggering exit at the very first off-ramp.
        """
        batch_size = small_batch["input_ids"].shape[0]
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=1.0,
            )
        counts = out["exit_counts"]
        assert counts[0] == batch_size, (
            f"With threshold=1.0, all docs must exit at first off-ramp. "
            f"exit_counts={counts}"
        )
        assert sum(counts[1:]) == 0, (
            f"With threshold=1.0, no docs should exit later. "
            f"exit_counts={counts}"
        )

    def test_threshold_zero_scores_match_baseline(self, model, small_batch):
        """With threshold=0.0, scores must equal the standard full-forward logit.

        Since no docs exit early, forward_compacted_early_exit runs all 6 layers
        and uses the same classifier, so scores must match forward_with_offramps
        final_logit.
        """
        with torch.no_grad():
            out_c = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.0,
            )
            out_ref = model.forward_with_offramps(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
            )
        assert torch.allclose(out_c["scores"], out_ref["final_logit"], atol=1e-4), (
            f"With threshold=0, scores must match final_logit. "
            f"Max diff: {(out_c['scores'] - out_ref['final_logit']).abs().max().item():.6f}"
        )

    def test_threshold_one_scores_match_offramp0(self, model, small_batch):
        """With threshold=1.0, scores must match off-ramp 0 logits."""
        with torch.no_grad():
            out_c = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=1.0,
            )
            out_ref = model.forward_with_offramps(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
            )
        offramp0_logits = out_ref["offramp_logits"][0]
        assert torch.allclose(out_c["scores"], offramp0_logits, atol=1e-4), (
            f"With threshold=1.0, scores must match off-ramp 0 logits. "
            f"Max diff: {(out_c['scores'] - offramp0_logits).abs().max().item():.6f}"
        )

    def test_exit_layer_consistent_with_exit_counts(self, model, small_batch):
        """exit_layer values must be consistent with exit_counts histogram."""
        with torch.no_grad():
            out = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=0.1,
            )
        el = out["exit_layer"]
        counts = out["exit_counts"]
        for layer_idx in range(6):
            expected = counts[layer_idx]
            actual = (el == layer_idx).sum().item()
            assert actual == expected, (
                f"exit_layer histogram mismatch at layer {layer_idx}: "
                f"exit_counts={expected}, histogram={actual}"
            )

    def test_scores_match_naive_early_exit(self, model, small_batch):
        """System C scores must match Baseline B for the same threshold.

        Both use the same exit criterion and the same off-ramp/classifier weights,
        so the final scores must be identical regardless of execution strategy
        (compacted vs. jagged batch).
        """
        threshold = 0.3
        with torch.no_grad():
            out_c = model.forward_compacted_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=threshold,
            )
            out_b = model.forward_naive_early_exit(
                small_batch["input_ids"],
                small_batch["attention_mask"],
                small_batch["token_type_ids"],
                entropy_threshold=threshold,
            )
        assert torch.allclose(out_c["scores"], out_b["scores"], atol=1e-4), (
            f"System C scores must match Baseline B scores for threshold={threshold}. "
            f"Max diff: {(out_c['scores'] - out_b['scores']).abs().max().item():.6f}"
        )
        assert torch.equal(out_c["exit_layer"], out_b["exit_layer"]), (
            f"System C exit_layer must match Baseline B exit_layer. "
            f"System C: {out_c['exit_layer'].tolist()}, "
            f"Baseline B: {out_b['exit_layer'].tolist()}"
        )


# ---------------------------------------------------------------------------
# Inference tests: run_system_c
# (require CUDA + pre-tokenized dev set)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def system_c_results():
    """Run System C once with a small batch and single threshold."""
    from src.inference import run_system_c

    return run_system_c(
        tokenized_path="data/dev_tokenized.pt",
        batch_size=8,
        thresholds=[0.1],
        output_dir="results",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    not os.path.exists("data/dev_tokenized.pt"),
    reason="Pre-tokenized dev set not found",
)
class TestRunSystemC:

    def test_returns_list(self, system_c_results):
        assert isinstance(system_c_results, list), "run_system_c must return a list"
        assert len(system_c_results) == 1, "Expected one result for one threshold"

    def test_result_keys(self, system_c_results):
        r = system_c_results[0]
        expected = {"system", "threshold", "mrr10", "mean_batch_latency_ms",
                    "total_latency_s", "batch_size", "exit_counts"}
        assert expected == set(r.keys()), (
            f"Missing or extra keys. Expected {expected}, got {set(r.keys())}"
        )

    def test_system_name(self, system_c_results):
        assert system_c_results[0]["system"] == "system_c"

    def test_threshold_recorded(self, system_c_results):
        assert system_c_results[0]["threshold"] == 0.1

    def test_mrr_in_range(self, system_c_results):
        mrr = system_c_results[0]["mrr10"]
        assert 0.0 < mrr <= 1.0, f"MRR@10 out of range: {mrr}"

    def test_latency_positive(self, system_c_results):
        r = system_c_results[0]
        assert r["mean_batch_latency_ms"] > 0
        assert r["total_latency_s"] > 0

    def test_exit_counts_sum(self, system_c_results):
        """exit_counts must sum to total number of (query, passage) pairs."""
        r = system_c_results[0]
        data = torch.load("data/dev_tokenized.pt", weights_only=False)
        total_pairs = data["input_ids"].shape[0]
        assert sum(r["exit_counts"]) == total_pairs, (
            f"exit_counts must sum to {total_pairs}, got {sum(r['exit_counts'])}"
        )

    def test_exit_counts_length(self, system_c_results):
        assert len(system_c_results[0]["exit_counts"]) == 6
