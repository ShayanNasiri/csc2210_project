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

        # Reference to the classification head
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
        # Run full BERT forward with intermediate hidden states
        outputs = self.backbone.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        # outputs.hidden_states: tuple of 7 tensors
        # Index 0 = embeddings, index 1-6 = after each transformer layer
        all_hidden = outputs.hidden_states

        offramp_logits = []
        offramp_entropies = []

        for i in range(5):  # off-ramps after layers 0-4
            logit = self.offramps(i, all_hidden[i + 1])
            entropy = self.offramps.ramps[i].compute_entropy(logit)
            offramp_logits.append(logit)
            offramp_entropies.append(entropy)

        # Final classifier on pooler output (after layer 5)
        final_logit = self.classifier(outputs.pooler_output).squeeze(-1)

        return {
            "final_logit": final_logit,
            "offramp_logits": offramp_logits,
            "offramp_entropies": offramp_entropies,
        }
