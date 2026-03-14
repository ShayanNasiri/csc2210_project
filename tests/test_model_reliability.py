"""Tests for model reliability: every assumption that could silently corrupt
baseline results if violated.

These tests verify that the EarlyExitCrossEncoder helper methods, exit-score
assignment, determinism, weight persistence, token structure, and shape
contracts all hold. A failure here means baseline A/B/C numbers cannot be
trusted.

Requires `transformers` — tests are skipped on systems without it.
"""

import math
import os

import pytest
import torch

from src.offramps import OffRampHead, OffRampCollection
from src.constants import NUM_BERT_LAYERS, NUM_OFFRAMPS, HIDDEN_SIZE


# ---------------------------------------------------------------------------
# Fixtures (require transformers — skip gracefully if unavailable)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model():
    pytest.importorskip("transformers")
    from src.model import EarlyExitCrossEncoder
    m = EarlyExitCrossEncoder()
    m.eval()
    return m


@pytest.fixture(scope="module")
def batch(model):
    """4 distinct query-passage pairs, max_length=32."""
    tokenizer = model.tokenizer
    queries = ["what is python", "best gpu", "neural network", "early exit"]
    passages = [
        "Python is a programming language.",
        "The RTX 4090 is a high-end GPU.",
        "Neural networks are inspired by the brain.",
        "Early exit reduces computation.",
    ]
    return tokenizer(
        queries, passages,
        max_length=32, padding="max_length", truncation=True,
        return_tensors="pt",
    )


@pytest.fixture(scope="module")
def single_pair(model):
    """Single query-passage pair (batch_size=1)."""
    tokenizer = model.tokenizer
    return tokenizer(
        ["what is python"],
        ["Python is a programming language."],
        max_length=32, padding="max_length", truncation=True,
        return_tensors="pt",
    )


# ---------------------------------------------------------------------------
# 1. Helper method shape contracts
# ---------------------------------------------------------------------------

class TestHelperShapes:

    def test_get_bert_embeddings_shape(self, model, batch):
        hidden = model._get_bert_embeddings(
            batch["input_ids"], batch["token_type_ids"]
        )
        B, S = batch["input_ids"].shape
        assert hidden.shape == (B, S, HIDDEN_SIZE), (
            f"Embeddings shape: expected ({B}, {S}, {HIDDEN_SIZE}), got {hidden.shape}"
        )

    def test_get_extended_attention_mask_shape(self, model, batch):
        mask = model._get_extended_attention_mask(
            batch["attention_mask"], batch["input_ids"].shape
        )
        B, S = batch["input_ids"].shape
        # BERT extended mask is (B, 1, 1, S) for self-attention
        assert mask.shape == (B, 1, 1, S), (
            f"Extended mask shape: expected ({B}, 1, 1, {S}), got {mask.shape}"
        )

    def test_apply_bert_layer_shape_preserved(self, model, batch):
        hidden = model._get_bert_embeddings(
            batch["input_ids"], batch["token_type_ids"]
        )
        ext_mask = model._get_extended_attention_mask(
            batch["attention_mask"], batch["input_ids"].shape
        )
        out = model._apply_bert_layer(0, hidden, ext_mask)
        assert out.shape == hidden.shape, (
            f"Layer output shape {out.shape} != input shape {hidden.shape}"
        )

    def test_apply_bert_layer_all_six_layers(self, model, batch):
        """Every layer from 0 to 5 produces valid output with preserved shape."""
        hidden = model._get_bert_embeddings(
            batch["input_ids"], batch["token_type_ids"]
        )
        ext_mask = model._get_extended_attention_mask(
            batch["attention_mask"], batch["input_ids"].shape
        )
        with torch.no_grad():
            for i in range(NUM_BERT_LAYERS):
                hidden = model._apply_bert_layer(i, hidden, ext_mask)
                B, S = batch["input_ids"].shape
                assert hidden.shape == (B, S, HIDDEN_SIZE), (
                    f"Layer {i} output shape wrong: {hidden.shape}"
                )

    def test_apply_bert_layer_returns_tensor_not_tuple(self, model, batch):
        """_apply_bert_layer handles the tuple-vs-tensor compatibility."""
        hidden = model._get_bert_embeddings(
            batch["input_ids"], batch["token_type_ids"]
        )
        ext_mask = model._get_extended_attention_mask(
            batch["attention_mask"], batch["input_ids"].shape
        )
        out = model._apply_bert_layer(0, hidden, ext_mask)
        assert isinstance(out, torch.Tensor), (
            f"Expected Tensor, got {type(out)} — compatibility wrapper may be broken"
        )

    def test_apply_pooler_and_classifier_shape(self, model, batch):
        with torch.no_grad():
            hidden = model._get_bert_embeddings(
                batch["input_ids"], batch["token_type_ids"]
            )
            ext_mask = model._get_extended_attention_mask(
                batch["attention_mask"], batch["input_ids"].shape
            )
            for i in range(NUM_BERT_LAYERS):
                hidden = model._apply_bert_layer(i, hidden, ext_mask)
            logits = model._apply_pooler_and_classifier(hidden)
        B = batch["input_ids"].shape[0]
        assert logits.shape == (B,), (
            f"Pooler+classifier output: expected ({B},), got {logits.shape}"
        )


# ---------------------------------------------------------------------------
# 2. batch_size=1 squeeze safety
# ---------------------------------------------------------------------------

class TestBatchSizeOne:

    def test_pooler_classifier_batch_one(self, model, single_pair):
        """squeeze(-1) must not collapse batch dim when batch=1."""
        with torch.no_grad():
            hidden = model._get_bert_embeddings(
                single_pair["input_ids"], single_pair["token_type_ids"]
            )
            ext_mask = model._get_extended_attention_mask(
                single_pair["attention_mask"], single_pair["input_ids"].shape
            )
            for i in range(NUM_BERT_LAYERS):
                hidden = model._apply_bert_layer(i, hidden, ext_mask)
            logits = model._apply_pooler_and_classifier(hidden)
        assert logits.shape == (1,), (
            f"batch=1: expected logits shape (1,), got {logits.shape}"
        )

    def test_offramp_head_batch_one(self):
        ramp = OffRampHead(hidden_size=HIDDEN_SIZE)
        x = torch.randn(1, 32, HIDDEN_SIZE)
        out = ramp(x)
        assert out.shape == (1,), (
            f"batch=1: expected off-ramp shape (1,), got {out.shape}"
        )

    def test_forward_naive_early_exit_batch_one(self, model, single_pair):
        with torch.no_grad():
            out = model.forward_naive_early_exit(
                single_pair["input_ids"],
                single_pair["attention_mask"],
                single_pair["token_type_ids"],
                entropy_threshold=0.1,
            )
        assert out["scores"].shape == (1,), (
            f"batch=1: expected scores (1,), got {out['scores'].shape}"
        )
        assert sum(out["exit_counts"]) == 1


# ---------------------------------------------------------------------------
# 3. Forward determinism
# ---------------------------------------------------------------------------

class TestForwardDeterminism:

    def test_naive_early_exit_deterministic(self, model, batch):
        """Two calls with identical input produce identical output (eval mode)."""
        with torch.no_grad():
            out1 = model.forward_naive_early_exit(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"], entropy_threshold=0.1,
            )
            out2 = model.forward_naive_early_exit(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"], entropy_threshold=0.1,
            )
        assert torch.equal(out1["scores"], out2["scores"]), (
            "Scores must be deterministic in eval mode"
        )
        assert torch.equal(out1["exit_layer"], out2["exit_layer"]), (
            "Exit layers must be deterministic in eval mode"
        )
        assert out1["exit_counts"] == out2["exit_counts"], (
            "Exit counts must be deterministic in eval mode"
        )

    def test_forward_with_offramps_deterministic(self, model, batch):
        with torch.no_grad():
            out1 = model.forward_with_offramps(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"],
            )
            out2 = model.forward_with_offramps(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"],
            )
        assert torch.equal(out1["final_logit"], out2["final_logit"])
        for i in range(NUM_OFFRAMPS):
            assert torch.equal(
                out1["offramp_logits"][i], out2["offramp_logits"][i]
            ), f"Off-ramp {i} logit not deterministic"


# ---------------------------------------------------------------------------
# 4. _zero_exited_docs must clone — no in-place mutation
# ---------------------------------------------------------------------------

class TestZeroExitedDocsNonMutation:

    def test_does_not_mutate_input(self, model):
        hidden = torch.randn(4, 16, HIDDEN_SIZE)
        original = hidden.clone()
        active_mask = torch.tensor([True, False, True, False])
        model._zero_exited_docs(hidden, active_mask)
        assert torch.equal(hidden, original), (
            "_zero_exited_docs must clone — it mutated the input tensor"
        )


# ---------------------------------------------------------------------------
# 5. Exit scores match off-ramp logits
# ---------------------------------------------------------------------------

class TestExitScoreCorrectness:

    def test_all_exit_at_ramp0_scores_match_offramp0(self, model, batch):
        """When threshold=1.0, all docs exit at ramp 0. Their scores must
        equal the off-ramp 0 logits from forward_with_offramps."""
        with torch.no_grad():
            ref = model.forward_with_offramps(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"],
            )
            out = model.forward_naive_early_exit(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"], entropy_threshold=1.0,
            )
        assert torch.allclose(out["scores"], ref["offramp_logits"][0], atol=1e-5), (
            f"Exit scores should equal off-ramp 0 logits. "
            f"Max diff: {(out['scores'] - ref['offramp_logits'][0]).abs().max().item():.6f}"
        )

    def test_no_exit_scores_match_final_classifier(self, model, batch):
        """When threshold=0.0, no exits. Scores must match the final classifier."""
        with torch.no_grad():
            ref = model.forward_with_offramps(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"],
            )
            out = model.forward_naive_early_exit(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"], entropy_threshold=0.0,
            )
        assert torch.allclose(out["scores"], ref["final_logit"], atol=1e-4), (
            f"No-exit scores must match final_logit. "
            f"Max diff: {(out['scores'] - ref['final_logit']).abs().max().item():.6f}"
        )


# ---------------------------------------------------------------------------
# 6. Off-ramp weight save/load roundtrip
# ---------------------------------------------------------------------------

class TestOfframpWeightPersistence:

    def test_save_load_roundtrip(self, model, batch, tmp_path):
        """Save off-ramp weights, load into a fresh model, verify identical outputs."""
        from src.model import EarlyExitCrossEncoder

        path = str(tmp_path / "offramp_weights.pt")

        # Save current weights
        torch.save(model.offramps.state_dict(), path)

        # Create fresh model and load weights
        fresh = EarlyExitCrossEncoder()
        fresh.eval()
        fresh.offramps.load_state_dict(torch.load(path, weights_only=True))

        with torch.no_grad():
            out_orig = model.forward_with_offramps(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"],
            )
            out_fresh = fresh.forward_with_offramps(
                batch["input_ids"], batch["attention_mask"],
                batch["token_type_ids"],
            )

        for i in range(NUM_OFFRAMPS):
            assert torch.equal(
                out_orig["offramp_logits"][i], out_fresh["offramp_logits"][i]
            ), f"Off-ramp {i} logit mismatch after load"

    def test_state_dict_has_expected_keys(self, model):
        """State dict must have 10 keys: weight + bias for each of 5 ramps."""
        sd = model.offramps.state_dict()
        assert len(sd) == 10, f"Expected 10 keys, got {len(sd)}: {list(sd.keys())}"
        for i in range(NUM_OFFRAMPS):
            assert f"ramps.{i}.linear.weight" in sd
            assert f"ramps.{i}.linear.bias" in sd

    def test_weight_shapes(self, model):
        sd = model.offramps.state_dict()
        for i in range(NUM_OFFRAMPS):
            w = sd[f"ramps.{i}.linear.weight"]
            b = sd[f"ramps.{i}.linear.bias"]
            assert w.shape == (1, HIDDEN_SIZE), f"Ramp {i} weight shape: {w.shape}"
            assert b.shape == (1,), f"Ramp {i} bias shape: {b.shape}"


# ---------------------------------------------------------------------------
# 7. Token type IDs are nontrivial
# ---------------------------------------------------------------------------

class TestTokenTypeIds:

    def test_tokenizer_produces_token_type_ids(self, batch):
        """The tokenizer must produce token_type_ids for BERT cross-encoder."""
        assert "token_type_ids" in batch, (
            "Tokenizer must return token_type_ids for query-passage pairs"
        )

    def test_token_type_ids_not_all_zero(self, batch):
        """For paired input, token_type_ids must have both 0s (query) and 1s (passage).
        If all zeros, the model treats everything as one segment — wrong for
        cross-encoder ranking."""
        tids = batch["token_type_ids"]
        has_zeros = (tids == 0).any().item()
        has_ones = (tids == 1).any().item()
        assert has_zeros and has_ones, (
            f"token_type_ids must have both 0 and 1 for paired input. "
            f"Unique values: {tids.unique().tolist()}"
        )

    def test_cls_token_at_index_zero(self, model, batch):
        """[CLS] token must be at index 0 — off-ramps extract hidden[:, 0, :]."""
        cls_id = model.tokenizer.cls_token_id
        assert (batch["input_ids"][:, 0] == cls_id).all(), (
            f"[CLS] token (id={cls_id}) must be at index 0. "
            f"Got: {batch['input_ids'][:, 0].tolist()}"
        )


# ---------------------------------------------------------------------------
# 8. forward_with_offramps hidden states structure
# ---------------------------------------------------------------------------

class TestHiddenStatesStructure:

    def test_hidden_states_count(self, model, batch):
        """Model must return 7 hidden states (embeddings + 6 layers) when
        output_hidden_states=True. Off-ramp training indexes into this."""
        with torch.no_grad():
            outputs = model.backbone.bert(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                output_hidden_states=True,
            )
        assert len(outputs.hidden_states) == NUM_BERT_LAYERS + 1, (
            f"Expected {NUM_BERT_LAYERS + 1} hidden states, "
            f"got {len(outputs.hidden_states)}"
        )

    def test_hidden_states_shapes(self, model, batch):
        """Each hidden state must be (B, S, H)."""
        B, S = batch["input_ids"].shape
        with torch.no_grad():
            outputs = model.backbone.bert(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                output_hidden_states=True,
            )
        for i, hs in enumerate(outputs.hidden_states):
            assert hs.shape == (B, S, HIDDEN_SIZE), (
                f"Hidden state {i} shape: expected ({B}, {S}, {HIDDEN_SIZE}), "
                f"got {hs.shape}"
            )


# ---------------------------------------------------------------------------
# 9. OffRampCollection boundary
# ---------------------------------------------------------------------------

class TestOffRampBoundary:

    def test_out_of_range_layer_idx_raises(self):
        """Accessing ramp 5 (only 0-4 exist) must raise IndexError."""
        collection = OffRampCollection(num_ramps=5, hidden_size=HIDDEN_SIZE)
        x = torch.randn(2, 16, HIDDEN_SIZE)
        with pytest.raises(IndexError):
            collection(5, x)
