import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.constants import MODEL_NAME, NUM_OFFRAMPS, HIDDEN_SIZE, NUM_BERT_LAYERS
from src.offramps import OffRampCollection
from src.triton_compact import compact_batch


class EarlyExitCrossEncoder(nn.Module):
    """Cross-encoder with off-ramp classifiers after layers 1-5."""

    def __init__(self, model_name: str = MODEL_NAME):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Freeze all backbone parameters
        for p in self.backbone.parameters():
            p.requires_grad = False

        # Reference to the classification head
        self.classifier = self.backbone.classifier

        # Off-ramps after layers 0 through NUM_OFFRAMPS-1
        self.offramps = OffRampCollection(num_ramps=NUM_OFFRAMPS, hidden_size=HIDDEN_SIZE)

    def _get_bert_embeddings(self, input_ids, token_type_ids=None):
        """Compute BERT embeddings for input sequence."""
        return self.backbone.bert.embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids
        )

    def _get_extended_attention_mask(self, attention_mask, input_shape):
        """Get extended attention mask for BERT layers."""
        return self.backbone.bert.get_extended_attention_mask(
            attention_mask, input_shape
        )

    def _apply_bert_layer(self, layer_idx, hidden_states, extended_mask):
        """Apply a single BERT layer with cross-version compatibility.

        Args:
            layer_idx: Index of layer to apply (0-5)
            hidden_states: (batch, seq_len, hidden_size)
            extended_mask: Extended attention mask

        Returns:
            Updated hidden_states of shape (batch, seq_len, hidden_size)
        """
        layer = self.backbone.bert.encoder.layer[layer_idx]
        out = layer(hidden_states, attention_mask=extended_mask)
        # Handle both tuple (cluster transformers) and tensor (local transformers) returns
        return out[0] if isinstance(out, tuple) else out

    def _apply_pooler_and_classifier(self, hidden_states):
        """Apply BERT pooler and final classification head.

        Args:
            hidden_states: (batch, seq_len, hidden_size)

        Returns:
            Logits of shape (batch,)
        """
        pooled = self.backbone.bert.pooler(hidden_states)
        return self.classifier(pooled).squeeze(-1)

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

        # Initialize embeddings and mask
        hidden_states = self._get_bert_embeddings(input_ids, token_type_ids)
        extended_mask = self._get_extended_attention_mask(attention_mask, input_ids.shape)

        # Initialize tracking tensors
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        scores = torch.zeros(batch_size, device=device)
        exit_layer = torch.zeros(batch_size, dtype=torch.long, device=device)
        exit_counts = [0] * (NUM_OFFRAMPS + 1)

        # Process through BERT layers with early-exit logic
        for i in range(NUM_BERT_LAYERS):
            hidden_states = self._apply_bert_layer(i, hidden_states, extended_mask)

            # Check off-ramp exit criterion after layers 0 through NUM_OFFRAMPS-1
            if i < NUM_OFFRAMPS:
                logit = self.offramps(i, hidden_states)
                entropy = self.offramps.ramps[i].compute_entropy(logit)
                newly_exited = active_mask & (entropy < entropy_threshold)
                if newly_exited.any():
                    scores[newly_exited] = logit[newly_exited].detach()
                    exit_layer[newly_exited] = i
                    exit_counts[i] += newly_exited.sum().item()
                    active_mask[newly_exited] = False
                    hidden_states = self._zero_exited_docs(hidden_states, active_mask)

        # Process remaining active docs through final classifier
        if active_mask.any():
            final_logits = self._apply_pooler_and_classifier(hidden_states)
            scores[active_mask] = final_logits[active_mask].detach()
            exit_layer[active_mask] = NUM_OFFRAMPS
            exit_counts[NUM_OFFRAMPS] = active_mask.sum().item()

        return {"scores": scores, "exit_layer": exit_layer, "exit_counts": exit_counts}

    def forward_compacted_early_exit(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        entropy_threshold: float = 0.1,
    ):
        """System C early-exit inference. Exited docs are physically removed from
        the batch via Triton compaction, eliminating wasted compute on padding.

        Returns dict with:
            scores:      (batch,) final relevance score for each doc
            exit_layer:  (batch,) int tensor, 0-4 = off-ramp index, 5 = full forward
            exit_counts: list of 6 ints — docs exiting at each point
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize embeddings and mask
        hidden_states = self._get_bert_embeddings(input_ids, token_type_ids)
        extended_mask = self._get_extended_attention_mask(attention_mask, input_ids.shape)

        # Track which original batch positions are still active
        global_indices = torch.arange(batch_size, device=device)

        # Initialize output tensors (full batch size)
        final_scores = torch.zeros(batch_size, device=device)
        exit_layer = torch.zeros(batch_size, dtype=torch.long, device=device)
        exit_counts = [0] * (NUM_OFFRAMPS + 1)

        for i in range(NUM_BERT_LAYERS):
            hidden_states = self._apply_bert_layer(i, hidden_states, extended_mask)

            if i < NUM_OFFRAMPS:
                logit = self.offramps(i, hidden_states)
                entropy = self.offramps.ramps[i].compute_entropy(logit)
                exited = entropy < entropy_threshold

                if exited.any():
                    # Record scores and exit info for exited items
                    final_scores[global_indices[exited]] = logit[exited].detach()
                    exit_layer[global_indices[exited]] = i
                    exit_counts[i] = exited.sum().item()

                    # Compact: physically remove exited rows
                    active_mask = ~exited
                    hidden_states, attention_mask, _ = compact_batch(
                        hidden_states, attention_mask, active_mask
                    )
                    global_indices = global_indices[active_mask]

                    # Recompute extended_mask for the new (smaller) batch
                    extended_mask = self._get_extended_attention_mask(
                        attention_mask, hidden_states.shape[:2]
                    )

                    # If no items remain, stop
                    if hidden_states.shape[0] == 0:
                        break

        # Process remaining active docs through final classifier
        if hidden_states.shape[0] > 0 and global_indices.shape[0] > 0:
            final_logits = self._apply_pooler_and_classifier(hidden_states)
            final_scores[global_indices] = final_logits.detach()
            exit_layer[global_indices] = NUM_OFFRAMPS
            exit_counts[NUM_OFFRAMPS] = global_indices.shape[0]

        return {"scores": final_scores, "exit_layer": exit_layer, "exit_counts": exit_counts}

    def _zero_exited_docs(self, hidden_states, active_mask):
        """Zero out hidden states for exited documents (jagged batch padding)."""
        hidden_states = hidden_states.clone()
        hidden_states[~active_mask] = 0.0
        return hidden_states

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

        for i in range(NUM_OFFRAMPS):  # off-ramps after layers 0-4
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
