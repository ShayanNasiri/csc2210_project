"""Shared utilities for inference runners."""

import torch
from typing import Dict, Tuple

from src.utils import TimerContext


def load_tokenized_data(path: str) -> Dict[str, torch.Tensor]:
    """Load pre-tokenized dev set.

    Returns:
        Dict with keys: input_ids, attention_mask, token_type_ids, qids, labels
    """
    data = torch.load(path, weights_only=False)
    return {
        "input_ids": data["input_ids"],
        "attention_mask": data["attention_mask"],
        "token_type_ids": data.get("token_type_ids"),
        "qids": data["qids"],
        "labels": data["labels"],
    }


def get_batch_slice(
    data: Dict[str, torch.Tensor],
    start_idx: int,
    end_idx: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract and transfer a batch slice to device.

    Returns:
        (input_ids, attention_mask, token_type_ids) on device
    """
    batch_ids = data["input_ids"][start_idx:end_idx].to(device)
    batch_mask = data["attention_mask"][start_idx:end_idx].to(device)
    batch_tids = (
        data["token_type_ids"][start_idx:end_idx].to(device)
        if data["token_type_ids"] is not None
        else None
    )
    return batch_ids, batch_mask, batch_tids


class BatchRunner:
    """Handles batch iteration with warmup and timing."""

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        warmup_batches: int = 10,
        timed_batch_limit: int = 100,
    ):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_batches = (num_samples + batch_size - 1) // batch_size
        self.warmup_batches = min(warmup_batches, self.num_batches)
        self.timed_batches = min(timed_batch_limit, self.num_batches)

    def warmup(
        self,
        data: Dict[str, torch.Tensor],
        device: torch.device,
        forward_fn,
    ):
        """Run warmup batches without timing."""
        for i in range(self.warmup_batches):
            start = i * self.batch_size
            end = min(start + self.batch_size, self.num_samples)
            batch_ids, batch_mask, batch_tids = get_batch_slice(data, start, end, device)
            forward_fn(batch_ids, batch_mask, batch_tids)

    def run_with_timing(
        self,
        data: Dict[str, torch.Tensor],
        device: torch.device,
        forward_fn,
    ) -> Tuple[list, list]:
        """Run all batches, timing the first `timed_batches`.

        Returns:
            (all_outputs, batch_latencies)
        """
        all_outputs = []
        batch_latencies = []

        for i in range(self.num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_ids, batch_mask, batch_tids = get_batch_slice(
                data, start_idx, end_idx, device
            )

            if i < self.timed_batches:
                timer = TimerContext(device)
                with timer:
                    output = forward_fn(batch_ids, batch_mask, batch_tids)
                batch_latencies.append(timer.elapsed_ms)
            else:
                output = forward_fn(batch_ids, batch_mask, batch_tids)

            all_outputs.append(output)

        return all_outputs, batch_latencies
