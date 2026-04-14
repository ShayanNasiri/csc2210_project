import argparse
import csv
import itertools
import os
import statistics

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification

from src.constants import (
    MODEL_NAME,
    NUM_OFFRAMPS,
    DEFAULT_DEV_DATA_PATH,
    DEFAULT_VAL_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_BATCH_SIZE,
    DEFAULT_ENTROPY_THRESHOLDS,
    DEFAULT_JOINT_WEIGHTS_PATH,
    DEFAULT_OFFRAMP_WEIGHTS_PATH,
    DEFAULT_SYSTEM_E_WEIGHTS_PATH,
    DEFAULT_PER_RAMP_GRID,
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
    results_tag: str = "",
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
    save_results(results, os.path.join(output_dir, f"{results_tag}baseline_a_results.json"))

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
    results_tag: str = "",
    weights_path: str | None = None,
) -> list:
    """Run naive early-exit inference (Baseline B) over a list of entropy thresholds.

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS
    if weights_path is None:
        weights_path = DEFAULT_OFFRAMP_WEIGHTS_PATH

    set_seed()
    device = get_device()

    # Load model + off-ramp weights
    model = EarlyExitCrossEncoder()
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
    save_results(all_results, os.path.join(output_dir, f"{results_tag}baseline_b_results.json"))

    return all_results


def run_system_c(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds: list | None = None,
    output_dir: str = DEFAULT_RESULTS_DIR,
    results_tag: str = "",
    weights_path: str | None = None,
) -> list:
    """Run Triton-compacted early-exit inference (System C) over a list of entropy thresholds.

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS
    if weights_path is None:
        weights_path = DEFAULT_OFFRAMP_WEIGHTS_PATH

    set_seed()
    device = get_device()

    # Load model + off-ramp weights
    model = EarlyExitCrossEncoder()
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
    save_results(all_results, os.path.join(output_dir, f"{results_tag}system_c_results.json"))

    return all_results


def run_system_d(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds: list | None = None,
    output_dir: str = DEFAULT_RESULTS_DIR,
    weights_path: str | None = None,
    results_tag: str = "",
) -> list:
    """Run System D: jointly-trained model with Triton-compacted early exit.

    Uses joint_weights.pt by default (backbone + offramps trained together with
    alpha=1.0). For the alpha sweep, pass a custom `weights_path` such as
    `results/joint_alpha0.5_weights.pt`.

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS

    set_seed()
    device = get_device()

    # Load model + joint weights (backbone + offramps)
    model = EarlyExitCrossEncoder()
    if weights_path is None:
        weights_path = os.path.join(output_dir, "joint_weights.pt")
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.backbone.load_state_dict(state["backbone"])
    model.offramps.load_state_dict(state["offramps"])
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
        def forward_fn(input_ids, attention_mask, token_type_ids, _t=threshold):
            return model.forward_compacted_early_exit(
                input_ids, attention_mask, token_type_ids, entropy_threshold=_t
            )

        with torch.no_grad():
            runner.warmup(data, device, forward_fn)
            all_outputs, batch_latencies = runner.run_with_timing(data, device, forward_fn)

        # Aggregate results
        all_scores = []
        global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
        for out in all_outputs:
            all_scores.extend(out["scores"].cpu().tolist())
            for j in range(NUM_OFFRAMPS + 1):
                global_exit_counts[j] += out["exit_counts"][j]

        total_timed_latency_ms = sum(batch_latencies)
        mean_batch_latency_ms = total_timed_latency_ms / len(batch_latencies)
        mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

        result = {
            "system": "system_d",
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
    save_results(all_results, os.path.join(output_dir, f"{results_tag}system_d_results.json"))

    return all_results


def run_system_e(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds: list | None = None,
    output_dir: str = DEFAULT_RESULTS_DIR,
    weights_path: str | None = None,
    results_tag: str = "",
) -> list:
    """Run System E: jointly-trained model with self-distillation (KL) weights.

    Uses system_e_joint_distill_weights.pt (backbone + offramps trained with
    KL distillation loss) instead of the standard joint_weights.pt.

    Args:
        weights_path: Path to weights file. If None, uses default System E path.

    Returns a list of result dicts, one per threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_ENTROPY_THRESHOLDS

    set_seed()
    device = get_device()

    # Load model + System E distillation weights (backbone + offramps)
    model = EarlyExitCrossEncoder()
    if weights_path is None:
        weights_path = os.path.join(output_dir, DEFAULT_SYSTEM_E_WEIGHTS_PATH.split("/")[-1])
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.backbone.load_state_dict(state["backbone"])
    model.offramps.load_state_dict(state["offramps"])
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
        def forward_fn(input_ids, attention_mask, token_type_ids, _t=threshold):
            return model.forward_compacted_early_exit(
                input_ids, attention_mask, token_type_ids, entropy_threshold=_t
            )

        with torch.no_grad():
            runner.warmup(data, device, forward_fn)
            all_outputs, batch_latencies = runner.run_with_timing(data, device, forward_fn)

        # Aggregate results
        all_scores = []
        global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
        for out in all_outputs:
            all_scores.extend(out["scores"].cpu().tolist())
            for j in range(NUM_OFFRAMPS + 1):
                global_exit_counts[j] += out["exit_counts"][j]

        total_timed_latency_ms = sum(batch_latencies)
        mean_batch_latency_ms = total_timed_latency_ms / len(batch_latencies)
        mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

        result = {
            "system": "system_e",
            "weights": os.path.basename(weights_path),
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
    save_results(all_results, os.path.join(output_dir, f"{results_tag}system_e_results.json"))

    return all_results


def run_per_ramp_threshold_sweep(
    task_id: int = 0,
    num_tasks: int = 49,
    grid_values: list | None = None,
    weights_path: str = "results/joint_alpha0.5_weights.pt",
    tokenized_path: str = DEFAULT_VAL_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: str = DEFAULT_RESULTS_DIR,
    sweep_subdir: str = "system_f_sweep_results",
    csv_prefix: str = "system_f",
    patience: int = 1,
) -> str:
    """Per-ramp entropy threshold grid sweep over a given weights file.

    Partitions the full grid_size**5 grid by (t0, t1) so each task processes
    grid_size**3 configs. With the default 7-value grid, num_tasks=49 (7x7) and
    each task evaluates 343 configs. task_id maps to (t0_idx, t1_idx) via
    integer division and modulo.

    Each task writes a CSV to
    ``<output_dir>/<sweep_subdir>/<csv_prefix>_t0=<t0>_t1=<t1>.csv``
    with one row per (t2, t3, t4) combination. The model, weights, and val
    data are loaded once per task; warmup runs once before the inner loop.

    The default ``sweep_subdir`` and ``csv_prefix`` reproduce the System F
    layout already committed under ``results/system_f_sweep_results/``. Other
    weight families (e.g. System G runs on System E β=1.0 weights) override
    both kwargs to land their CSVs in an isolated subdirectory with a
    distinctive filename prefix, so a re-run can never overwrite committed
    System F outputs.
    """
    if grid_values is None:
        grid_values = DEFAULT_PER_RAMP_GRID

    grid_size = len(grid_values)
    expected_tasks = grid_size * grid_size
    if num_tasks != expected_tasks:
        raise ValueError(
            f"num_tasks must be {expected_tasks} for grid size {grid_size}, "
            f"got {num_tasks}"
        )
    if not (0 <= task_id < num_tasks):
        raise ValueError(f"task_id must be in [0, {num_tasks}), got {task_id}")

    t0_idx = task_id // grid_size
    t1_idx = task_id % grid_size
    t0 = grid_values[t0_idx]
    t1 = grid_values[t1_idx]

    set_seed()
    device = get_device()

    # Load model + System D alpha=0.5 weights
    model = EarlyExitCrossEncoder()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.backbone.load_state_dict(state["backbone"])
    model.offramps.load_state_dict(state["offramps"])
    model.to(device)
    model.eval()

    # Load val data once
    data = load_tokenized_data(tokenized_path)
    num_samples = data["input_ids"].shape[0]

    runner = BatchRunner(
        num_samples=num_samples,
        batch_size=batch_size,
        warmup_batches=WARMUP_BATCHES,
        timed_batch_limit=TIMED_BATCH_LIMIT,
    )

    # Output directory + CSV path with distinctive (t0, t1) filename.
    # sweep_subdir and csv_prefix are parameterized so the same driver can
    # serve System F (default), System G (System E β=1.0 weights), or any
    # future weight family without colliding output paths.
    sweep_dir = os.path.join(output_dir, sweep_subdir)
    os.makedirs(sweep_dir, exist_ok=True)
    csv_path = os.path.join(sweep_dir, f"{csv_prefix}_t0={t0}_t1={t1}.csv")

    fieldnames = [
        "t0", "t1", "t2", "t3", "t4",
        "patience",
        "mrr10", "mean_batch_latency_ms",
        "exit_count_0", "exit_count_1", "exit_count_2",
        "exit_count_3", "exit_count_4", "exit_count_5",
    ]

    # Warmup once with a representative threshold vector
    warmup_thresholds = [t0, t1, grid_values[0], grid_values[0], grid_values[0]]

    def warmup_fn(input_ids, attention_mask, token_type_ids, _t=warmup_thresholds):
        return model.forward_compacted_early_exit(
            input_ids, attention_mask, token_type_ids, entropy_threshold=_t,
            patience=patience,
        )

    with torch.no_grad():
        runner.warmup(data, device, warmup_fn)

    # Inner sub-grid: (t2, t3, t4) — grid_size**3 configs
    sub_grid = list(itertools.product(grid_values, repeat=3))
    print(
        f"[task {task_id}/{num_tasks}] t0={t0} t1={t1} — "
        f"{len(sub_grid)} configs -> {csv_path}"
    )

    rows_written = 0
    flush_every = 25

    # lineterminator="\n" overrides csv.writer's default "\r\n" so the
    # cluster-written CSVs are byte-identical to the LF-normalized copies
    # checked into git. Without this, every cluster sweep produces 49 CSVs
    # that look "modified" in `git status` purely from CRLF vs LF endings.
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        f.flush()

        for t2, t3, t4 in sub_grid:
            thresholds_vec = [t0, t1, t2, t3, t4]

            def forward_fn(
                input_ids, attention_mask, token_type_ids, _t=thresholds_vec
            ):
                return model.forward_compacted_early_exit(
                    input_ids, attention_mask, token_type_ids, entropy_threshold=_t,
                    patience=patience,
                )

            with torch.no_grad():
                all_outputs, batch_latencies = runner.run_with_timing(
                    data, device, forward_fn
                )

            all_scores = []
            global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
            for out in all_outputs:
                all_scores.extend(out["scores"].cpu().tolist())
                for j in range(NUM_OFFRAMPS + 1):
                    global_exit_counts[j] += out["exit_counts"][j]

            mean_lat = sum(batch_latencies) / len(batch_latencies)
            mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

            row = {
                "t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4,
                "patience": patience,
                "mrr10": mrr10,
                "mean_batch_latency_ms": mean_lat,
                "exit_count_0": global_exit_counts[0],
                "exit_count_1": global_exit_counts[1],
                "exit_count_2": global_exit_counts[2],
                "exit_count_3": global_exit_counts[3],
                "exit_count_4": global_exit_counts[4],
                "exit_count_5": global_exit_counts[5],
            }
            writer.writerow(row)
            rows_written += 1

            if rows_written % flush_every == 0:
                f.flush()
                print(
                    f"  [{rows_written}/{len(sub_grid)}] "
                    f"({t2}, {t3}, {t4}) MRR={mrr10:.4f} "
                    f"lat={mean_lat:.2f}ms"
                )

        f.flush()

    print(f"[task {task_id}] DONE — {rows_written} rows -> {csv_path}")
    return csv_path


def _run_per_ramp_single_point(
    system_name: str,
    weights_path: str,
    thresholds,
    patience: int = 1,
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    output_dir: str = DEFAULT_RESULTS_DIR,
    results_tag: str = "",
) -> dict:
    """Shared backend for the System F / G / H single-operating-point runners.

    Evaluates ONE configuration (a length-5 per-ramp threshold vector plus a
    patience value) and writes ``{results_tag}{system_name}_results.json``.
    The JSON is a one-element list to match the Baseline A / System D / E
    schema (those schemas are lists-over-thresholds; a single operating point
    is simply a list of length 1).
    """
    if not hasattr(thresholds, "__len__") or len(thresholds) != 5:
        raise ValueError(
            f"thresholds must have length 5 (one per ramp), got "
            f"{getattr(thresholds, '__len__', lambda: '?')()}"
        )
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    thresholds = [float(t) for t in thresholds]

    set_seed()
    device = get_device()

    model = EarlyExitCrossEncoder()
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.backbone.load_state_dict(state["backbone"])
    model.offramps.load_state_dict(state["offramps"])
    model.to(device)
    model.eval()

    data = load_tokenized_data(tokenized_path)
    num_samples = data["input_ids"].shape[0]

    runner = BatchRunner(
        num_samples=num_samples,
        batch_size=batch_size,
        warmup_batches=WARMUP_BATCHES,
        timed_batch_limit=TIMED_BATCH_LIMIT,
    )

    def forward_fn(input_ids, attention_mask, token_type_ids,
                   _t=list(thresholds), _p=patience):
        return model.forward_compacted_early_exit(
            input_ids, attention_mask, token_type_ids,
            entropy_threshold=_t, patience=_p,
        )

    with torch.no_grad():
        runner.warmup(data, device, forward_fn)
        all_outputs, batch_latencies = runner.run_with_timing(data, device, forward_fn)

    all_scores = []
    global_exit_counts = [0] * (NUM_OFFRAMPS + 1)
    for out in all_outputs:
        all_scores.extend(out["scores"].cpu().tolist())
        for j in range(NUM_OFFRAMPS + 1):
            global_exit_counts[j] += out["exit_counts"][j]

    total_ms = sum(batch_latencies)
    mean_ms = total_ms / len(batch_latencies)
    mrr10 = compute_mrr_at_k(data["qids"], all_scores, data["labels"], k=10)

    result = {
        "system": system_name,
        "weights": os.path.basename(weights_path),
        "thresholds": list(thresholds),
        "patience": patience,
        "mrr10": mrr10,
        "mean_batch_latency_ms": mean_ms,
        "total_latency_s": total_ms / 1000.0,
        "batch_size": batch_size,
        "exit_counts": global_exit_counts,
    }

    print(
        f"{system_name} thresholds={thresholds} patience={patience} — "
        f"MRR@10: {mrr10:.4f}, Latency: {mean_ms:.2f} ms, "
        f"Exit counts: {global_exit_counts}"
    )

    os.makedirs(output_dir, exist_ok=True)
    save_results(
        [result],
        os.path.join(output_dir, f"{results_tag}{system_name}_results.json"),
    )
    return result


def run_system_f(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds=None,
    output_dir: str = DEFAULT_RESULTS_DIR,
    weights_path: str | None = None,
    results_tag: str = "",
) -> dict:
    """System F: System D α=0.5 weights + per-ramp entropy threshold vector.

    Inference-only single-operating-point variant. Caller must supply a
    length-5 ``thresholds`` list; patience is fixed at 1 (System F is PABEE
    patience=1 by definition).
    """
    if weights_path is None:
        weights_path = os.path.join(output_dir, "joint_alpha0.5_weights.pt")
    return _run_per_ramp_single_point(
        system_name="system_f",
        weights_path=weights_path,
        thresholds=thresholds if thresholds is not None else [],
        patience=1,
        tokenized_path=tokenized_path,
        batch_size=batch_size,
        output_dir=output_dir,
        results_tag=results_tag,
    )


def run_system_g(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds=None,
    output_dir: str = DEFAULT_RESULTS_DIR,
    weights_path: str | None = None,
    results_tag: str = "",
) -> dict:
    """System G: System E β=1.0 weights + per-ramp entropy threshold vector.

    Inference-only single-operating-point variant. Patience is fixed at 1.
    """
    if weights_path is None:
        weights_path = os.path.join(output_dir, "system_e_beta1.0_weights.pt")
    return _run_per_ramp_single_point(
        system_name="system_g",
        weights_path=weights_path,
        thresholds=thresholds if thresholds is not None else [],
        patience=1,
        tokenized_path=tokenized_path,
        batch_size=batch_size,
        output_dir=output_dir,
        results_tag=results_tag,
    )


def run_system_h(
    tokenized_path: str = DEFAULT_DEV_DATA_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    thresholds=None,
    patience: int = 2,
    output_dir: str = DEFAULT_RESULTS_DIR,
    weights_path: str | None = None,
    results_tag: str = "",
) -> dict:
    """System H: System E β=1.0 weights + per-ramp thresholds + PABEE patience ≥ 2.

    Inference-only. ``patience`` must be ≥ 2 — P=1 is System G, not System H.
    """
    if patience < 2:
        raise ValueError(
            f"System H requires patience >= 2 (P=1 is System G, not System H); "
            f"got patience={patience}"
        )
    if weights_path is None:
        weights_path = os.path.join(output_dir, "system_e_beta1.0_weights.pt")
    return _run_per_ramp_single_point(
        system_name="system_h",
        weights_path=weights_path,
        thresholds=thresholds if thresholds is not None else [],
        patience=patience,
        tokenized_path=tokenized_path,
        batch_size=batch_size,
        output_dir=output_dir,
        results_tag=results_tag,
    )


def _parse_thresholds_arg(s: str) -> list:
    """Parse a ``--thresholds`` CLI string ``"t0,t1,t2,t3,t4"`` into a list of 5 floats."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            f"--thresholds must be 5 comma-separated floats, got {len(parts)}: {s!r}"
        )
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--thresholds must be 5 comma-separated floats: {e}"
        )


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
                weights_path = DEFAULT_OFFRAMP_WEIGHTS_PATH
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
    parser.add_argument("--weights_path", type=str, default=None)
    parser.add_argument("--results_tag", type=str, default="",
                        help="Prefix for result filenames, e.g. 'val_' or 'test_'")
    parser.add_argument("--task_id", type=int, default=0,
                        help="Array task ID for per_ramp_sweep (0-indexed)")
    parser.add_argument("--num_tasks", type=int, default=49,
                        help="Total number of array tasks for per_ramp_sweep")
    parser.add_argument("--sweep_subdir", type=str, default="system_f_sweep_results",
                        help="Subdirectory under --output_dir for per_ramp_sweep CSVs")
    parser.add_argument("--csv_prefix", type=str, default="system_f",
                        help="Filename prefix for per_ramp_sweep CSVs (before _t0=...)")
    parser.add_argument("--patience", type=int, default=1,
                        help="PABEE patience P: exit after P consecutive below-threshold ramps")
    parser.add_argument("--thresholds", type=_parse_thresholds_arg, default=None,
                        help="5 comma-separated floats (per-ramp thresholds) "
                             "for system_f/system_g/system_h single-point runs")
    args = parser.parse_args()

    if args.system == "baseline_a":
        run_baseline_a(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            results_tag=args.results_tag,
        )
    elif args.system == "baseline_b":
        run_baseline_b(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "system_c":
        run_system_c(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "system_d":
        run_system_d(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "system_e":
        run_system_e(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "system_f":
        run_system_f(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            thresholds=args.thresholds,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "system_g":
        run_system_g(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            thresholds=args.thresholds,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "system_h":
        run_system_h(
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            thresholds=args.thresholds,
            patience=args.patience,
            output_dir=args.output_dir,
            weights_path=args.weights_path,
            results_tag=args.results_tag,
        )
    elif args.system == "full_sweep":
        run_full_sweep(
            tokenized_path=args.data_path,
            output_dir=args.output_dir,
        )
    elif args.system == "per_ramp_sweep":
        run_per_ramp_threshold_sweep(
            task_id=args.task_id,
            num_tasks=args.num_tasks,
            weights_path=(
                args.weights_path
                if args.weights_path is not None
                else "results/joint_alpha0.5_weights.pt"
            ),
            tokenized_path=args.data_path,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
            sweep_subdir=args.sweep_subdir,
            csv_prefix=args.csv_prefix,
            patience=args.patience,
        )
