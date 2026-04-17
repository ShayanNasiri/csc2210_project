"""Generate figures for the CSC2210 project report.

Reads the final test-set JSONs under results/test_set_results/ and produces:
  (1) pareto.pdf  — MRR@10 vs mean batch latency for all 8 systems, plus an
                    MRR = MRR(A) line and "Pareto dominated" shading.
  (2) exits.pdf   — normalized exit-layer distribution for the three inference-
                    tuned systems (F / G / H P=2) side-by-side.

Only depends on matplotlib + numpy; no ML stack required.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "results" / "test_set_results"
OUT_DIR = ROOT / "docs" / "Project_Report"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _first(x):
    """Return the first element of a list or the raw dict."""
    return x[0] if isinstance(x, list) else x


def load_final_points():
    """Return a list of dicts: {label, mrr, latency, exits (optional)}.

    Picks the single operating point that appears in the paper narrative.
    For B/C (where the single threshold collapses everything to layer 1), we
    use the first threshold entry — they are all MRR-identical anyway.
    For D/E we use threshold=0.01 (the single-threshold winner reported in
    CLAUDE.md / PROJECT_HISTORY).
    """
    def _load(path: Path):
        with path.open() as f:
            return json.load(f)

    a = _load(FINAL_DIR / "test_final_baseline_a_results.json")
    b = _first(_load(FINAL_DIR / "test_final_baseline_b_results.json"))
    c = _first(_load(FINAL_DIR / "test_final_system_c_results.json"))

    d_list = _load(FINAL_DIR / "test_final_system_d_results.json")
    d = next(r for r in d_list if abs(r["threshold"] - 0.01) < 1e-9)

    e_list = _load(FINAL_DIR / "test_final_system_e_results.json")
    e = next(r for r in e_list if abs(r["threshold"] - 0.01) < 1e-9)

    f = _first(_load(FINAL_DIR / "test_final_system_f_results.json"))
    g = _first(_load(FINAL_DIR / "test_final_system_g_results.json"))
    h = _first(_load(FINAL_DIR / "test_final_system_h_results.json"))

    points = [
        {"label": "A", "mrr": a["mrr10"], "lat": a["mean_batch_latency_ms"]},
        {"label": "B", "mrr": b["mrr10"], "lat": b["mean_batch_latency_ms"],
         "exits": b["exit_counts"]},
        {"label": "C", "mrr": c["mrr10"], "lat": c["mean_batch_latency_ms"],
         "exits": c["exit_counts"]},
        {"label": "D", "mrr": d["mrr10"], "lat": d["mean_batch_latency_ms"],
         "exits": d["exit_counts"]},
        {"label": "E", "mrr": e["mrr10"], "lat": e["mean_batch_latency_ms"],
         "exits": e["exit_counts"]},
        {"label": "F", "mrr": f["mrr10"], "lat": f["mean_batch_latency_ms"],
         "exits": f["exit_counts"]},
        {"label": "G", "mrr": g["mrr10"], "lat": g["mean_batch_latency_ms"],
         "exits": g["exit_counts"]},
        {"label": "H", "mrr": h["mrr10"], "lat": h["mean_batch_latency_ms"],
         "exits": h["exit_counts"]},
    ]
    return points


# ---------------------------------------------------------------------------
# Figure 1: Pareto frontier
# ---------------------------------------------------------------------------

LONG_NAMES = {
    "A": "Baseline A",
    "B": "Baseline B",
    "C": "System C",
    "D": r"System D",
    "E": r"System E",
    "F": "System F",
    "G": "System G",
    "H": r"System H ($P{=}2$)",
}

# Manual label offsets so text never lands on top of a marker.
# (dx_points, dy_points) — relative to marker in display space.
LABEL_OFFSETS = {
    "A": (-23, -15),
    "B": (-50, -3),
    "C": (8, -3),
    "D": (-1, -14),
    "E": (-45, -2),
    "F": (-43, -4),
    "G": (-20, 12),
    "H": (6, -2),
}


def plot_pareto(points, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))

    lats = np.array([p["lat"] for p in points])
    mrrs = np.array([p["mrr"] for p in points])

    # Classify: champion G, baselines, others.
    categories = {
        "Baseline A (target)": ["A"],
        "Naive / frozen-ramp": ["B", "C"],
        "Joint-trained (single $\\tau$)": ["D", "E"],
        "Per-ramp $\\mathbf{\\tau}$": ["F", "G", "H"],
    }
    style = {
        "Baseline A (target)": dict(marker="*", s=220, color="#cc3333", zorder=5),
        "Naive / frozen-ramp": dict(marker="X", s=95, color="#888888", zorder=3),
        "Joint-trained (single $\\tau$)": dict(marker="o", s=80, color="#1f77b4", zorder=4),
        "Per-ramp $\\mathbf{\\tau}$": dict(marker="s", s=80, color="#2ca02c", zorder=4),
    }

    for group, labels in categories.items():
        xs = [p["lat"] for p in points if p["label"] in labels]
        ys = [p["mrr"] for p in points if p["label"] in labels]
        ax.scatter(xs, ys, label=group, edgecolor="black", linewidth=0.6, **style[group])

    # Highlight champion G.
    g = next(p for p in points if p["label"] == "G")
    ax.scatter([g["lat"]], [g["mrr"]], s=250, facecolor="none",
               edgecolor="#2ca02c", linewidth=2.0, zorder=6)

    # Dashed reference line at Baseline A's MRR.
    a = next(p for p in points if p["label"] == "A")
    ax.axhline(a["mrr"], color="#cc3333", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.text(ax.get_xlim()[1] * 0.02 + 5, a["mrr"] + 0.015,
            "MRR@10 = Baseline A", color="#cc3333", fontsize=8)

    # Annotate each point.
    for p in points:
        dx, dy = LABEL_OFFSETS[p["label"]]
        ax.annotate(LONG_NAMES[p["label"]],
                    xy=(p["lat"], p["mrr"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=8)

    ax.set_xlabel("Mean batch latency (ms, batch=64, RTX 4090)")
    ax.set_ylabel("Test MRR@10 (MS MARCO dev)")
    ax.set_xlim(3, 42)
    ax.set_ylim(-0.02, 0.82)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", fontsize=7.5, framealpha=0.95,
              bbox_to_anchor=(0.5, -0.20), ncol=4,
              columnspacing=1.0, handletextpad=0.4, borderpad=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: exit distribution
# ---------------------------------------------------------------------------


def plot_exits(points, out_path: Path):
    """Exit distribution across all 8 systems.

    Baselines A/B/C have degenerate exit distributions (100% at one layer);
    we render them with hatched, grey bars and a smaller bar width so the eye
    reads them as the degenerate reference, not as peers to D-H.

    Baseline A never fires an off-ramp (no early-exit rule), so by convention
    we put 100% of its mass at the 'Final (L6)' column.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # Synthesize Baseline A's "exit distribution": all docs go through the
    # full 6 layers, so the distribution is 100% at the Final column.
    a = next(p for p in points if p["label"] == "A").copy()
    a["exits"] = [0, 0, 0, 0, 0, 62357]

    trained = ["D", "E", "F", "G", "H"]
    baselines = ["A", "B", "C"]

    names = {
        "A": "A (full model)",
        "B": "B (naive)",
        "C": "C (Triton, frozen)",
        "D": r"D",
        "E": r"E",
        "F": "F",
        "G": "G (champion)",
        "H": r"H $P{=}2$",
    }
    trained_colors = {"D": "#1f77b4", "E": "#17becf", "F": "#2ca02c",
                      "G": "#ff7f0e", "H": "#9467bd"}
    # Baselines get distinct colors (not greyscale) so each is readable on its
    # own, while hatching + smaller bar width still signal "degenerate".
    baseline_colors = {"A": "#cc3333",  # crimson — matches Baseline A star in Fig 1
                       "B": "#8c564b",  # brown
                       "C": "#e377c2"}  # pink

    x = np.arange(6)  # ramps 0..4 + final(5)

    # --- Baselines A/B/C: grouped on the LEFT of each tick, hatched grey ---
    bw_b = 0.10
    for i, lbl in enumerate(baselines):
        p = a if lbl == "A" else next(pp for pp in points if pp["label"] == lbl)
        counts = np.array(p["exits"], dtype=float)
        frac = counts / counts.sum()
        offset = (i - 1) * bw_b - 0.32  # clustered on the left side of each tick
        ax.bar(x + offset, frac, width=bw_b, label=names[lbl],
               color=baseline_colors[lbl], hatch="//", edgecolor="black",
               linewidth=0.4, alpha=0.85)

    # --- Joint-trained systems D/E/F/G/H: grouped on the RIGHT, solid ---
    bw_t = 0.11
    for i, lbl in enumerate(trained):
        p = next(pp for pp in points if pp["label"] == lbl)
        counts = np.array(p["exits"], dtype=float)
        frac = counts / counts.sum()
        offset = (i - 2) * bw_t + 0.13  # clustered right of each tick
        ax.bar(x + offset, frac, width=bw_t, label=names[lbl],
               color=trained_colors[lbl], edgecolor="black", linewidth=0.4)

    # Vertical separators between ticks to visually group the two clusters.
    for xc in x[:-1]:
        ax.axvline(xc + 0.5, color="#cccccc", linewidth=0.5, zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(["Ramp 0", "Ramp 1", "Ramp 2", "Ramp 3", "Ramp 4",
                        "Final (L6)"], fontsize=8)
    ax.set_ylabel("Fraction of docs exiting at layer")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper center", fontsize=7.2, ncol=4, framealpha=0.95,
              bbox_to_anchor=(0.5, 1.22), columnspacing=1.0,
              handletextpad=0.4, borderpad=0.4)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    points = load_final_points()
    for p in points:
        print(f"  {p['label']:>2}: MRR {p['mrr']:.4f}  lat {p['lat']:.2f} ms")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_pareto(points, OUT_DIR / "pareto.pdf")
    plot_pareto(points, OUT_DIR / "pareto.png")
    plot_exits(points, OUT_DIR / "exits.pdf")
    plot_exits(points, OUT_DIR / "exits.png")


if __name__ == "__main__":
    main()
