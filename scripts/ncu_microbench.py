"""Isolated micro-benchmark for Nsight Compute profiling of compact_batch kernel.

Uses dummy tensors only — does NOT load the full dataset.
Finishes in under 10 seconds without Nsight. Nsight replays kernels internally.
"""
import torch
from src.triton_compact import compact_batch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy tensors mimicking a batch mid-inference
    B, S, H = 256, 128, 384
    hidden_states = torch.randn(B, S, H, device=device, dtype=torch.float32)
    attention_mask = torch.ones(B, S, device=device, dtype=torch.long)
    # ~50% active
    torch.manual_seed(42)
    active_mask = torch.rand(B, device=device) > 0.5

    # Warmup
    for _ in range(5):
        compact_batch(hidden_states, attention_mask, active_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Profiled region — Nsight will capture this single call
    compacted_h, compacted_a, _ = compact_batch(hidden_states, attention_mask, active_mask)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"compact_batch: {B} -> {compacted_h.shape[0]} active rows")
    print("Microbench complete.")


if __name__ == "__main__":
    main()
