import argparse
import os

import torch
from transformers import AutoModelForSequenceClassification

from src.evaluate import compute_mrr_at_k, save_results
from src.utils import TimerContext, get_device, set_seed


def run_baseline_a(
    tokenized_path: str = "data/dev_tokenized.pt",
    batch_size: int = 64,
    output_dir: str = "results",
) -> dict:
    """Run standard cross-encoder inference (Baseline A) on pre-tokenized dev set."""
    set_seed()
    device = get_device()

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    model.to(device)
    model.eval()

    # Load pre-tokenized data
    data = torch.load(tokenized_path, weights_only=False)
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    qids = data["qids"]
    labels = data["labels"]

    num_samples = input_ids.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_scores = []

    with torch.no_grad():
        # Warmup: 10 batches
        warmup_batches = min(10, num_batches)
        for i in range(warmup_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch_ids = input_ids[start:end].to(device)
            batch_mask = attention_mask[start:end].to(device)
            model(input_ids=batch_ids, attention_mask=batch_mask)

        # Timed inference over all batches
        timed_batches = min(100, num_batches)
        batch_latencies = []

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_ids = input_ids[start_idx:end_idx].to(device)
            batch_mask = attention_mask[start_idx:end_idx].to(device)

            if i < timed_batches:
                timer = TimerContext(device)
                with timer:
                    outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
                batch_latencies.append(timer.elapsed_ms)
            else:
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)

            logits = outputs.logits.squeeze(-1).cpu()
            all_scores.extend(logits.tolist())

    # Compute total latency (sum of timed batches)
    total_timed_latency_ms = sum(batch_latencies)
    mean_batch_latency_ms = total_timed_latency_ms / len(batch_latencies)

    # Compute MRR@10
    mrr10 = compute_mrr_at_k(qids, all_scores, labels, k=10)

    results = {
        "system": "baseline_a",
        "mrr10": mrr10,
        "mean_batch_latency_ms": mean_batch_latency_ms,
        "total_latency_s": total_timed_latency_ms / 1000.0,
        "batch_size": batch_size,
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    save_results(results, os.path.join(output_dir, "baseline_a_results.json"))

    print(f"Baseline A — MRR@10: {mrr10:.4f}")
    print(f"Mean batch latency: {mean_batch_latency_ms:.2f} ms")
    print(f"Total timed latency: {total_timed_latency_ms / 1000.0:.2f} s")
    print(f"Batch size: {batch_size}, Batches: {num_batches}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="baseline_a")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="data/dev_tokenized.pt")
    args = parser.parse_args()

    if args.system == "baseline_a":
        run_baseline_a(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
        )
