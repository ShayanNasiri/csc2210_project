import argparse
import os
import statistics

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from src.constants import (
    MODEL_NAME,
    NUM_OFFRAMPS,
    DEFAULT_DEV_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENTROPY_THRESHOLDS,
    WARMUP_BATCHES,
    TIMED_BATCH_LIMIT,
)
from src.evaluate import compute_mrr_at_k, save_results
from src.inference_utils import load_tokenized_data, BatchRunner
from src.model import EarlyExitCrossEncoder
from src.utils import get_device, set_seed


def run_baseline_a(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: str = DEFAULT_RESULTS_DIR,
) -> dict:
    """Run standard cross-encoder inference (Baseline A) on pre-tokenized dev set."""
    set_seed()
    device = get_device()

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    # Load data
    data = load_tokenized_data(tokenized_path)
    num_samples = data["input_ids"].shape[0]

    # Set up batch runner
    runner = BatchRunner(
        num_samples=num_samples,
        batch_size=batch_size,
        warmup_batches=WARMUP_BATCHES,
        timed_batch_limit=TIMED_BATCH_LIMIT,
    )

    # Define forward function for standard inference
    def forward_fn(input_ids, attention_mask, token_type_ids):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

    with torch.no_grad():
        # Warmup
        runner.warmup(data, device, forward_fn)

        # Timed inference
        all_outputs, batch_latencies = runner.run_with_timing(data, device, forward_fn)

    # Extract scores
    all_scores = []
    for outputs in all_outputs:
        logits = outputs.logits.squeeze(-1).cpu()
        all_scores.extend(logits.tolist())

    # Compute metrics
    total_timed_latency_ms = sum(batch_latencies)
    mean_batch_latency_ms = total_timed_latency_ms / len(batch_latencies)
    mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

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
    print(f"Batch size: {batch_size}, Batches: {runner.num_batches}")

    return results


def run_baseline_b(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds: list | None = None,
    output_dir: str = DEFAULT_RESULTS_DIR,
) -> list:
    """Run naive early-exit inference (Baseline B) over a list of entropy thresholds.

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS

    set_seed()
    device = get_device()

    # Load model + off-ramp weights
    model = EarlyExitCrossEncoder()
    weights_path = os.path.join(output_dir, "offramp_weights.pt")
    model.offramps.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Load data
    data = load_tokenized_data(tokenized_path)
    num_samples = data["input_ids"].shape[0]

    # Set up batch runner
    runner = BatchRunner(
        num_samples=num_samples,
        batch_size=batch_size,
        warmup_batches=WARMUP_BATCHES,
        timed_batch_limit=TIMED_BATCH_LIMIT,
    )

    all_results = []

    for threshold in thresholds:
        # Define forward function with current threshold
        def forward_fn(input_ids, attention_mask, token_type_ids):
            return model.forward_naive_early_exit(
                input_ids, attention_mask, token_type_ids, entropy_threshold=threshold
            )

        with torch.no_grad():
            # Warmup
            runner.warmup(data, device, forward_fn)

            # Timed inference
            all_outputs, batch_latencies = runner.run_with_timing(data, device, forward_fn)

        # Aggregate results
        all_scores = []
        global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
        for out in all_outputs:
            all_scores.extend(out["scores"].cpu().tolist())
            for j in range(NUM_OFFRAMPS + 1):
                global_exit_counts[j] += out["exit_counts"][j]

        # Compute metrics
        total_timed_latency_ms = sum(batch_latencies)
        mean_batch_latency_ms = total_timed_latency_ms / len(batch_latencies)
        mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

        result = {
            "system": "baseline_b",
            "threshold": threshold,
            "mrr10": mrr10,
            "mean_batch_latency_ms": mean_batch_latency_ms,
            "total_latency_s": total_timed_latency_ms / 1000.0,
            "batch_size": batch_size,
            "exit_counts": global_exit_counts,
        }
        all_results.append(result)

        print(
            f"Threshold {threshold:.2f} — MRR@10: {mrr10:.4f}, "
            f"Latency: {mean_batch_latency_ms:.2f} ms, "
            f"Exit counts: {global_exit_counts}"
        )

    os.makedirs(output_dir, exist_ok=True)
    save_results(all_results, os.path.join(output_dir, "baseline_b_results.json"))

    return all_results


def run_system_c(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds: list | None = None,
    output_dir: str = DEFAULT_RESULTS_DIR,
) -> list:
    """Run Triton-compacted early-exit inference (System C) over a list of entropy thresholds.

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS

    set_seed()
    device = get_device()

    # Load model + off-ramp weights
    model = EarlyExitCrossEncoder()
    weights_path = os.path.join(output_dir, "offramp_weights.pt")
    model.offramps.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    # Load data
    data = load_tokenized_data(tokenized_path)
    num_samples = data["input_ids"].shape[0]

    # Set up batch runner
    runner = BatchRunner(
        num_samples=num_samples,
        batch_size=batch_size,
        warmup_batches=WARMUP_BATCHES,
        timed_batch_limit=TIMED_BATCH_LIMIT,
    )

    all_results = []

    for threshold in thresholds:
        # Define forward function with current threshold
        def forward_fn(input_ids, attention_mask, token_type_ids):
            return model.forward_compacted_early_exit(
                input_ids, attention_mask, token_type_ids, entropy_threshold=threshold
            )

        with torch.no_grad():
            # Warmup
            runner.warmup(data, device, forward_fn)

            # Timed inference
            all_outputs, batch_latencies = runner.run_with_timing(data, device, forward_fn)

        # Aggregate results
        all_scores = []
        global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
        for out in all_outputs:
            all_scores.extend(out["scores"].cpu().tolist())
            for j in range(NUM_OFFRAMPS + 1):
                global_exit_counts[j] += out["exit_counts"][j]

        # Compute metrics
        total_timed_latency_ms = sum(batch_latencies)
        mean_batch_latency_ms = total_timed_latency_ms / len(batch_latencies)
        mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

        result = {
            "system": "system_c",
            "threshold": threshold,
            "mrr10": mrr10,
            "mean_batch_latency_ms": mean_batch_latency_ms,
            "total_latency_s": total_timed_latency_ms / 1000.0,
            "batch_size": batch_size,
            "exit_counts": global_exit_counts,
        }
        all_results.append(result)

        print(
            f"Threshold {threshold:.2f} — MRR@10: {mrr10:.4f}, "
            f"Latency: {mean_batch_latency_ms:.2f} ms, "
            f"Exit counts: {global_exit_counts}"
        )

    os.makedirs(output_dir, exist_ok=True)
    save_results(all_results, os.path.join(output_dir, "system_c_results.json"))

    return all_results


DEFAULT_SWEEP_SYSTEMS = ["baseline_a", "baseline_b", "system_c"]
DEFAULT_SWEEP_BATCH_SIZES = [32, 64, 128, 256, 512]


def _exit_counts_to_pct(exit_counts, total):
    """Convert exit_counts list to per-layer pct dict."""
    if total == 0:
        return {f"pct_exit_layer{i}": 0.0 for i in range(NUM_OFFRAMPS)} | {"pct_exit_final": 0.0}
    return (
        {f"pct_exit_layer{i}": exit_counts[i] / total for i in range(NUM_OFFRAMPS)}
        | {"pct_exit_final": exit_counts[NUM_OFFRAMPS] / total}
    )


def run_full_sweep(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    systems: list | None = None,
    batch_sizes: list | None = None,
    thresholds: list | None = None,
    output_dir: str = DEFAULT_RESULTS_DIR,
) -> str:
    """Run all three systems at multiple batch sizes and thresholds.

    Saves results/full_sweep.csv and returns its path.
    """
    if systems is None:
        systems = DEFAULT_SWEEP_SYSTEMS
    if batch_sizes is None:
        batch_sizes = DEFAULT_SWEEP_BATCH_SIZES
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS

    set_seed()
    device = get_device()

    data = load_tokenized_data(tokenized_path)
    num_samples = data["input_ids"].shape[0]

    rows = []

    for system in systems:
        for batch_size in batch_sizes:
            print(f"=== system={system} batch_size={batch_size} ===")

            # Build model for this system
            if system == "baseline_a":
                model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                model.to(device)
                model.eval()
            else:
                model = EarlyExitCrossEncoder()
                weights_path = os.path.join(output_dir, "offramp_weights.pt")
                model.offramps.load_state_dict(
                    torch.load(weights_path, map_location=device, weights_only=True)
                )
                model.to(device)
                model.eval()

            runner = BatchRunner(
                num_samples=num_samples,
                batch_size=batch_size,
                warmup_batches=WARMUP_BATCHES,
                timed_batch_limit=TIMED_BATCH_LIMIT,
            )

            # Baseline A: single run (no threshold)
            if system == "baseline_a":
                def fwd_a(input_ids, attention_mask, token_type_ids):
                    return model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                    )

                with torch.no_grad():
                    runner.warmup(data, device, fwd_a)
                    all_outputs, latencies = runner.run_with_timing(data, device, fwd_a)

                all_scores = []
                for out in all_outputs:
                    all_scores.extend(out.logits.squeeze(-1).cpu().tolist())

                mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)
                mean_lat = sum(latencies) / len(latencies)
                std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

                row = {
                    "system": "baseline_a",
                    "batch_size": batch_size,
                    "threshold": float("nan"),
                    "mrr10": mrr10,
                    "mean_latency_ms": mean_lat,
                    "std_latency_ms": std_lat,
                }
                row.update({f"pct_exit_layer{i}": 0.0 for i in range(NUM_OFFRAMPS)})
                row["pct_exit_final"] = 1.0
                rows.append(row)
                print(f"  MRR={mrr10:.4f}, Latency={mean_lat:.2f}ms")

            # Baseline B / System C: sweep thresholds
            else:
                fwd_map = {
                    "baseline_b": model.forward_naive_early_exit,
                    "system_c": model.forward_compacted_early_exit,
                }
                fwd_method = fwd_map[system]

                for threshold in thresholds:
                    def forward_fn(input_ids, attention_mask, token_type_ids, _t=threshold):
                        return fwd_method(
                            input_ids, attention_mask, token_type_ids,
                            entropy_threshold=_t,
                        )

                    with torch.no_grad():
                        runner.warmup(data, device, forward_fn)
                        all_outputs, latencies = runner.run_with_timing(
                            data, device, forward_fn
                        )

                    all_scores = []
                    global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
                    for out in all_outputs:
                        all_scores.extend(out["scores"].cpu().tolist())
                        for j in range(NUM_OFFRAMPS + 1):
                            global_exit_counts[j] += out["exit_counts"][j]

                    mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)
                    mean_lat = sum(latencies) / len(latencies)
                    std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

                    row = {
                        "system": system,
                        "batch_size": batch_size,
                        "threshold": threshold,
                        "mrr10": mrr10,
                        "mean_latency_ms": mean_lat,
                        "std_latency_ms": std_lat,
                    }
                    row.update(_exit_counts_to_pct(global_exit_counts, num_samples))
                    rows.append(row)
                    print(f"  threshold={threshold:.2f} MRR={mrr10:.4f}, Latency={mean_lat:.2f}ms")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "full_sweep.csv")
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")
    return csv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default="baseline_a")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--data_path", type=str, default="data/dev_tokenized.pt")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    if args.system == "baseline_a":
        run_baseline_a(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    elif args.system == "baseline_b":
        run_baseline_b(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    elif args.system == "system_c":
        run_system_c(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
    elif args.system == "full_sweep":
        run_full_sweep(
            tokenized_path=args.data_path,
            output_dir=args.output_dir,
        )
