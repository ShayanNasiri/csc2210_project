"""Generate Pareto trade-off plots from full_sweep.csv.

Run locally after fetching results/full_sweep.csv from the cluster:
    python scripts/plot_pareto.py

Outputs 4 PNGs to results/:
    pareto_frontier.png
    speedup_vs_batchsize.png
    exit_distribution.png
    accuracy_degradation.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLORS = {
    "baseline_a": "tab:blue",
    "baseline_b": "tab:orange",
    "system_c": "tab:green",
}


def load_sweep(csv_path: str = "results/full_sweep.csv") -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Run `sbatch scripts/run_profiling.sh` on the cluster first, "
            "then fetch with: scp snasiri@cs.toronto.edu:~/csc2210_project/results/full_sweep.csv results/"
        )
    return pd.read_csv(csv_path)


def plot_pareto_frontier(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 1: Pareto frontier — latency vs MRR@10, one subplot per batch size."""
    batch_sizes = sorted(df["batch_size"].unique())
    fig, axes = plt.subplots(1, len(batch_sizes), figsize=(20, 4), sharey=True)

    for ax, bs in zip(axes, batch_sizes):
        sub = df[df["batch_size"] == bs]
        for system, grp in sub.groupby("system"):
            grp_sorted = grp.sort_values("mean_latency_ms")
            ax.plot(
                grp_sorted["mean_latency_ms"],
                grp_sorted["mrr10"],
                marker="o",
                label=system,
                color=COLORS.get(system),
            )
        ax.set_title(f"batch_size={bs}")
        ax.set_xlabel("Mean Latency (ms)")
        if ax is axes[0]:
            ax.set_ylabel("MRR@10")
        ax.legend()

    plt.suptitle("Pareto Frontier: Latency vs MRR@10", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_frontier.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def plot_speedup_vs_batchsize(df: pd.DataFrame, output_dir: str, threshold: float = 0.1) -> None:
    """Plot 2: Speedup of System C over Baseline B as a function of batch size."""
    b_lat = (
        df[(df["system"] == "baseline_b") & (df["threshold"] == threshold)]
        .set_index("batch_size")["mean_latency_ms"]
    )
    c_lat = (
        df[(df["system"] == "system_c") & (df["threshold"] == threshold)]
        .set_index("batch_size")["mean_latency_ms"]
    )
    speedup = b_lat / c_lat

    plt.figure(figsize=(8, 4))
    speedup.sort_index().plot(marker="o", color=COLORS["system_c"])
    plt.axhline(1, color="gray", linestyle="--", alpha=0.5)
    plt.xlabel("Batch Size")
    plt.ylabel(f"Speedup (Baseline B / System C)\nthreshold={threshold}")
    plt.title("Compaction Speedup vs Batch Size")
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "speedup_vs_batchsize.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def plot_exit_distribution(df: pd.DataFrame, output_dir: str, batch_size: int = 64) -> None:
    """Plot 3: Stacked bar chart of exit layer distribution for System C."""
    exit_cols = sorted(
        [c for c in df.columns if c.startswith("pct_exit_layer")]
    ) + ["pct_exit_final"]

    sub = df[(df["system"] == "system_c") & (df["batch_size"] == batch_size)].copy()
    sub = sub.set_index("threshold")[exit_cols]

    sub.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.xlabel("Entropy Threshold")
    plt.ylabel("Fraction of Documents")
    plt.title(f"Exit Layer Distribution (System C, batch_size={batch_size})")
    plt.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    plt.tight_layout()
    path = os.path.join(output_dir, "exit_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved {path}")
    plt.close()


def plot_accuracy_degradation(df: pd.DataFrame, output_dir: str) -> None:
    """Plot 4: MRR@10 degradation (%) vs entropy threshold."""
    a_mrr = df[df["system"] == "baseline_a"]["mrr10"].mean()
    print(f"Baseline A MRR@10: {a_mrr:.4f}")

    sub = df[df["system"].isin(["baseline_b", "system_c"])].copy()
    sub["mrr_degradation_pct"] = (a_mrr - sub["mrr10"]) / a_mrr * 100

    plt.figure(figsize=(8, 4))
    for system, grp in sub.groupby("system"):
        grp_sorted = grp.sort_values("threshold")
        plt.plot(
            grp_sorted["threshold"],
            grp_sorted["mrr_degradation_pct"],
            marker="o",
            label=system,
            color=COLORS.get(system),
        )
    plt.axhline(5, color="red", linestyle="--", label="5% limit")
    plt.xlabel("Entropy Threshold")
    plt.ylabel("MRR@10 Degradation (%)")
    plt.title("Accuracy Degradation vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_degradation.png")
    plt.savefig(path, dpi=150)
    print(f"Saved {path}")
    plt.close()


def main():
    sns.set_theme(style="whitegrid")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    df = load_sweep()
    print(f"Loaded {len(df)} rows from full_sweep.csv")

    plot_pareto_frontier(df, output_dir)
    plot_speedup_vs_batchsize(df, output_dir)
    plot_exit_distribution(df, output_dir)
    plot_accuracy_degradation(df, output_dir)

    print("\nAll plots saved to results/")


if __name__ == "__main__":
    main()
