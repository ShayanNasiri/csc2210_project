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

    def forward_naive_early_exit(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        entropy_threshold: float = 0.1,
    ):
        """Naive early-exit inference. Exited docs are zeroed between layers
        but stay in the batch (demonstrating the jagged batch problem).

        Returns dict with:
            scores:      (batch,) final relevance score for each doc
            exit_layer:  (batch,) int tensor, 0-4 = off-ramp index, 5 = full forward
            exit_counts: list of 6 ints — docs exiting at each point
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Embeddings
        hidden_states = self.backbone.bert.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )
        extended_mask = self.backbone.bert.get_extended_attention_mask(
            attention_mask, input_ids.shape
        )

        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        scores = torch.zeros(batch_size, device=device)
        exit_layer = torch.zeros(batch_size, dtype=torch.long, device=device)
        exit_counts = [0] * 6

        layers = self.backbone.bert.encoder.layer

        for i in range(6):
            out = layers[i](hidden_states, attention_mask=extended_mask)
            hidden_states = out[0] if isinstance(out, tuple) else out

            if i < 5:  # off-ramp after layers 0-4
                logit = self.offramps(i, hidden_states)  # (batch,)
                entropy = self.offramps.ramps[i].compute_entropy(logit)

                newly_exited = active_mask & (entropy < entropy_threshold)
                if newly_exited.any():
                    scores[newly_exited] = logit[newly_exited].detach()
                    exit_layer[newly_exited] = i
                    exit_counts[i] += newly_exited.sum().item()
                    active_mask[newly_exited] = False
                    hidden_states = hidden_states.clone()
                    hidden_states[~active_mask] = 0.0

        # Remaining active docs go through the pooler + classifier
        if active_mask.any():
            pooled = self.backbone.bert.pooler(hidden_states)
            final_logits = self.classifier(pooled).squeeze(-1)  # (batch,)
            scores[active_mask] = final_logits[active_mask].detach()
            exit_layer[active_mask] = 5
            exit_counts[5] = active_mask.sum().item()

        return {"scores": scores, "exit_layer": exit_layer, "exit_counts": exit_counts}

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
