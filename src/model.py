import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.offramps import OffRampCollection


class EarlyExitCrossEncoder(nn.Module):
    """Cross-encoder with off-ramp classifiers after layers 1-5."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # References to internal components
        self.embeddings = self.backbone.bert.embeddings
        self.layers = self.backbone.bert.encoder.layer
        self.pooler = self.backbone.bert.pooler
        self.classifier = self.backbone.classifier

        # 5 off-ramps: after layers 0-4 (1-indexed: layers 1-5)
        self.offramps = OffRampCollection(num_ramps=5, hidden_size=384)

    def forward_with_offramps(self, input_ids, attention_mask, token_type_ids=None):
        """Run all layers, collecting off-ramp logits and entropies.

        Returns dict with:
            final_logit: (batch,) from the standard classifier head
            offramp_logits: list of 5 tensors, each (batch,)
            offramp_entropies: list of 5 tensors, each (batch,)
        """
        # Get embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )

        # Build extended attention mask (same as BERT encoder expects)
        # Shape: (batch, 1, 1, seq_len) for broadcasting in attention layers
        extended_mask = self.backbone.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        offramp_logits = []
        offramp_entropies = []

        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states, attention_mask=extended_mask)[0]

            if i < 5:  # off-ramps after layers 0-4
                logit = self.offramps(i, hidden_states)
                entropy = self.offramps.ramps[i].compute_entropy(logit)
                offramp_logits.append(logit)
                offramp_entropies.append(entropy)

        # Final classifier: pooler (dense + tanh on [CLS]) → classifier
        pooled = self.pooler(hidden_states)
        final_logit = self.classifier(pooled).squeeze(-1)

        return {
            "final_logit": final_logit,
            "offramp_logits": offramp_logits,
            "offramp_entropies": offramp_entropies,
        }
