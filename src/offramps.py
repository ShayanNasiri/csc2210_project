import torch
import torch.nn as nn

from src.constants import HIDDEN_SIZE, NUM_OFFRAMPS


class OffRampHead(nn.Module):
    """Single off-ramp classifier: Linear(hidden_size, 1) on the [CLS] token."""

    def __init__(self, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Extract [CLS] token, classify, return logit of shape (batch,)."""
        cls_hidden = hidden_states[:, 0, :]  # (batch, hidden)
        return self.linear(cls_hidden).squeeze(-1)  # (batch,)

    def compute_entropy(self, logit: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of sigmoid output. Input/output shape: (batch,)."""
        p = torch.sigmoid(logit)
        p = torch.clamp(p, min=1e-7, max=1 - 1e-7)
        return -p * torch.log(p) - (1 - p) * torch.log(1 - p)


class OffRampCollection(nn.Module):
    """Collection of off-ramp heads (one after each early BERT layer)."""

    def __init__(self, num_ramps: int = NUM_OFFRAMPS, hidden_size: int = HIDDEN_SIZE):
        super().__init__()
        self.ramps = nn.ModuleList(
            [OffRampHead(hidden_size) for _ in range(num_ramps)]
        )

    def forward(self, layer_idx: int, hidden_states: torch.Tensor) -> torch.Tensor:
        """Route to the correct off-ramp by layer index."""
        return self.ramps[layer_idx](hidden_states)
