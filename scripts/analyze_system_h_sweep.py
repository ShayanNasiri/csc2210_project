"""Analyze the System H patience-based early-exit sweep results.

System H = System E (alpha=1.0, beta=1.0) weights + per-ramp threshold
vector + PABEE patience P >= 2. Each task of the 196-task sweep fixes
(P, t0, t1) and sweeps (t2, t3, t4) over 7^3 = 343 configurations.

This script analyzes ONE patience band at a time (selected via
``--patience``). Running it twice with ``--patience 2`` and
``--patience 3`` gives two independent reports — exactly what the
patience-vs-MRR comparison needs.

Loads CSVs from ``results/system_h_sweep_results/system_h_p<P>_t0=*_t1=*.csv``,
applies the 16.5 ms latency cap, ranks configs by MRR@10, reports the
Pareto frontier, and counts configs that strictly dominate each prior
champion (System D alpha=0.5, System F winner, System G winner).

Usage::

    python scripts/analyze_system_h_sweep.py --patience 2
    python scripts/analyze_system_h_sweep.py --patience 3
    python scripts/analyze_system_h_sweep.py --patience 4
    python scripts/analyze_system_h_sweep.py --patience 5

The script is fully reproducible — no GPU, no model load, no network.

Anchoring constants:
    CAP         = 16.5 ms (latency ceiling for promotion)
    CHAMP_MRR   = 0.3638 (System D alpha=0.5)
    CHAMP_LAT   = 16.14 ms
    SYSF_MRR    = 0.4688 (System F winner)
    SYSF_LAT    = 16.38 ms
    SYSG_MRR    = 0.5580 (System G winner — current champion)
    SYSG_LAT    = 16.48 ms
    BA_MRR      = 0.7380 (Baseline A target)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os


HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "system_h_sweep_results"))

CAP = 16.5
CHAMP_MRR = 0.3638
CHAMP_LAT = 16.14
SYSF_MRR = 0.4688
SYSF_LAT = 16.38
SYSG_MRR = 0.5580
SYSG_LAT = 16.48
BA_MRR = 0.7380

VALID_PATIENCES = (2, 3, 4, 5)


def load_rows(data_dir: str, patience: int) -> list[dict]:
    """Load all per-ramp sweep rows for a given patience band.

    Filters by filename prefix ``system_h_p<P>_*`` — the patience is
    encoded in the filename by the sweep driver.
    """
    rows: list[dict] = []
    pattern = os.path.join(data_dir, f"system_h_p{patience}_t0=*_t1=*.csv")
    for fn in sorted(glob.glob(pattern)):
        with open(fn) as f:
            for row in csv.DictReader(f):
                rows.append({
                    "t": (
                        float(row["t0"]),
                        float(row["t1"]),
                        float(row["t2"]),
                        float(row["t3"]),
                        float(row["t4"]),
                    ),
                    "mrr": float(row["mrr10"]),
                    "lat": float(row["mean_batch_latency_ms"]),
                    "ec": [int(row[f"exit_count_{i}"]) for i in range(6)],
                })
    return rows


def configs_under_cap(rows: list[dict], cap: float) -> list[dict]:
    """Rows with ``lat <= cap``, sorted by MRR descending."""
    out = [r for r in rows if r["lat"] <= cap]
    out.sort(key=lambda r: -r["mrr"])
    return out


def pareto_frontier(rows: list[dict]) -> list[dict]:
    """Pareto frontier on (lat asc, MRR desc). Returns points sorted by latency."""
    cand = sorted(rows, key=lambda r: (r["lat"], -r["mrr"]))
    front: list[dict] = []
    best_mrr = -1.0
    for r in cand:
        if r["mrr"] > best_mrr:
            front.append(r)
            best_mrr = r["mrr"]
    return front


def strict_dominators(rows: list[dict], ref_mrr: float, ref_lat: float) -> list[dict]:
    """Rows with ``mrr > ref_mrr`` AND ``lat < ref_lat`` (both strict)."""
    return [r for r in rows if r["mrr"] > ref_mrr and r["lat"] < ref_lat]


def _print_report(rows: list[dict], patience: int) -> None:
    expected = 7 ** 5
    print(f"data dir: {DATA_DIR}")
    print(f"patience P={patience}")
    print(f"total configs loaded: {len(rows)} / expected {expected}")
    if len(rows) != expected:
        print(f"WARNING: incomplete sweep — {expected - len(rows)} configs missing")

    under = configs_under_cap(rows, CAP)

    print()
    print(f"=== Top 25 configs with latency <= {CAP} ms (P={patience}) ===")
    header = (
        f"{'rank':>4}  "
        f"{'t0':>6} {'t1':>6} {'t2':>6} {'t3':>6} {'t4':>6}  "
        f"{'MRR':>7}  {'lat':>7}  "
        f"{'ec0':>6} {'ec1':>6} {'ec2':>6} {'ec3':>6} {'ec4':>6} {'ec5':>6}"
    )
    print(header)
    for i, r in enumerate(under[:25]):
        t = r["t"]
        ec = r["ec"]
        print(
            f"{i+1:>4}  "
            f"{t[0]:>6} {t[1]:>6} {t[2]:>6} {t[3]:>6} {t[4]:>6}  "
            f"{r['mrr']:>7.4f}  {r['lat']:>7.2f}  "
            f"{ec[0]:>6} {ec[1]:>6} {ec[2]:>6} {ec[3]:>6} {ec[4]:>6} {ec[5]:>6}"
        )

    if not under:
        print("(no configs under cap)")
        return

    winner = under[0]

    def _compare(label: str, ref_mrr: float, ref_lat: float) -> None:
        d_mrr = winner["mrr"] - ref_mrr
        d_lat = winner["lat"] - ref_lat
        print()
        print(f"=== Winner vs {label} ===")
        print(
            f"Winner MRR: {winner['mrr']:.4f}  "
            f"({label}: {ref_mrr:.4f})  "
            f"delta: {d_mrr:+.4f}  ({100*d_mrr/ref_mrr:+.1f}%)"
        )
        print(
            f"Winner lat: {winner['lat']:.2f} ms  "
            f"({label}: {ref_lat:.2f} ms)  "
            f"delta: {d_lat:+.2f} ms"
        )

    _compare("System D alpha=0.5", CHAMP_MRR, CHAMP_LAT)
    _compare("System F winner", SYSF_MRR, SYSF_LAT)
    _compare("System G winner (current champion)", SYSG_MRR, SYSG_LAT)

    if winner["mrr"] > SYSG_MRR:
        print(f">>> SCENARIO A: P={patience} beats System G — candidate for new champion!")
    elif winner["mrr"] > SYSF_MRR:
        print(f">>> SCENARIO B: P={patience} beats System F but not System G")
    elif winner["mrr"] > CHAMP_MRR:
        print(f">>> SCENARIO C: P={patience} beats System D alpha=0.5 but not System F")
    else:
        print(f">>> SCENARIO D: P={patience} does NOT improve over System D alpha=0.5")

    gap = 100 * (BA_MRR - winner["mrr"]) / BA_MRR
    print()
    print(f"Winner (P={patience}) is {gap:.1f}% below Baseline A ({BA_MRR:.4f})")

    overall = max(rows, key=lambda r: r["mrr"])
    print()
    print("=== Overall max MRR (ignoring cap) ===")
    print(
        f"  thresholds = {overall['t']}  "
        f"MRR={overall['mrr']:.4f}  lat={overall['lat']:.2f} ms"
    )

    print()
    print(f"=== Pareto frontier (under cap, P={patience}, sorted by latency) ===")
    front = pareto_frontier(under)
    print(f"{len(front)} points on frontier")
    for r in front:
        print(f"  t={r['t']}  MRR={r['mrr']:.4f}  lat={r['lat']:.2f} ms")

    for label, ref_mrr, ref_lat in (
        ("System D champion", CHAMP_MRR, CHAMP_LAT),
        ("System F winner", SYSF_MRR, SYSF_LAT),
        ("System G winner", SYSG_MRR, SYSG_LAT),
    ):
        dom = strict_dominators(rows, ref_mrr, ref_lat)
        dom.sort(key=lambda r: -r["mrr"])
        print()
        print(
            f"=== configs that strictly dominate {label} "
            f"(MRR > {ref_mrr} AND lat < {ref_lat}) ==="
        )
        print(f"count: {len(dom)}")
        for r in dom[:15]:
            print(f"  t={r['t']}  MRR={r['mrr']:.4f}  lat={r['lat']:.2f} ms")

    print()
    print(f"=== MRR distribution under cap (P={patience}) ===")
    buckets = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.3638),
        (0.3638, 0.4),
        (0.4, 0.4688),
        (0.4688, 0.5),
        (0.5, 0.5580),
        (0.5580, 0.6),
        (0.6, 0.7),
    ]
    for lo, hi in buckets:
        n = sum(1 for r in under if lo <= r["mrr"] < hi)
        print(f"  [{lo:.4f}, {hi:.4f}): {n:>5}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--patience", type=int, required=True, choices=VALID_PATIENCES,
        help="Patience band to analyze (P in {2, 3, 4, 5})",
    )
    parser.add_argument(
        "--data_dir", type=str, default=DATA_DIR,
        help=f"Directory containing System H sweep CSVs (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    rows = load_rows(args.data_dir, args.patience)
    _print_report(rows, args.patience)


if __name__ == "__main__":
    main()
