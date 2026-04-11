"""Analyze the System F per-ramp threshold sweep results.

Loads all 49 CSVs from results/system_f_sweep_results/, applies the
16.5 ms latency cap, ranks configurations by MRR@10, and prints the
top configs, the Pareto frontier, the count of configs that strictly
dominate the System D alpha=0.5 champion, and the MRR distribution.

Run from anywhere:
    python scripts/analyze_system_f_sweep.py

The script is fully reproducible — no GPU, no model load, no network.
It just reads the 49 CSVs in results/system_f_sweep_results/ and
recomputes the rankings deterministically.

Constants the analysis is anchored to:
    CAP        = 16.5 ms (latency ceiling for promotion)
    CHAMP_MRR  = 0.3638 (System D alpha=0.5 champion MRR)
    CHAMP_LAT  = 16.14 ms (System D alpha=0.5 champion latency)
    BA_MRR     = 0.7380 (Baseline A target MRR, no early exit)
"""
import csv
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(HERE, "..", "results", "system_f_sweep_results"))

CAP = 16.5
CHAMP_MRR = 0.3638
CHAMP_LAT = 16.14
BA_MRR = 0.7380


def load_rows(data_dir: str) -> list[dict]:
    rows: list[dict] = []
    pattern = os.path.join(data_dir, "system_f_t0=*.csv")
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


def main() -> None:
    rows = load_rows(DATA_DIR)
    print(f"data dir: {DATA_DIR}")
    print(f"total configs loaded: {len(rows)} / expected {7**5}")
    if len(rows) != 7**5:
        print("WARNING: incomplete sweep — fewer than 16,807 configs found")

    under = [r for r in rows if r["lat"] <= CAP]
    under.sort(key=lambda r: -r["mrr"])

    print()
    print(f"=== Top 25 configs with latency <= {CAP} ms ===")
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

    winner = under[0]
    print()
    print("=== Winner vs System D alpha=0.5 champion ===")
    d_mrr = winner["mrr"] - CHAMP_MRR
    d_lat = winner["lat"] - CHAMP_LAT
    print(
        f"Winner MRR: {winner['mrr']:.4f}  "
        f"(champion: {CHAMP_MRR:.4f})  "
        f"delta: {d_mrr:+.4f}  ({100*d_mrr/CHAMP_MRR:+.1f}%)"
    )
    print(
        f"Winner lat: {winner['lat']:.2f} ms  "
        f"(champion: {CHAMP_LAT:.2f} ms)  "
        f"delta: {d_lat:+.2f} ms"
    )
    gap_winner = 100 * (BA_MRR - winner["mrr"]) / BA_MRR
    gap_champ = 100 * (BA_MRR - CHAMP_MRR) / BA_MRR
    print(
        f"Winner is {gap_winner:.1f}% below Baseline A ({BA_MRR:.4f}); "
        f"champion was {gap_champ:.1f}% below"
    )

    overall = max(rows, key=lambda r: r["mrr"])
    print()
    print("=== Overall max MRR (ignoring cap) ===")
    print(
        f"  thresholds = {overall['t']}  "
        f"MRR={overall['mrr']:.4f}  lat={overall['lat']:.2f} ms"
    )

    print()
    print("=== Pareto frontier (under cap only, sorted by latency) ===")
    pareto = []
    cand = sorted(under, key=lambda r: (r["lat"], -r["mrr"]))
    best_mrr_seen = -1.0
    for r in cand:
        if r["mrr"] > best_mrr_seen:
            pareto.append(r)
            best_mrr_seen = r["mrr"]
    print(f"{len(pareto)} points on frontier")
    for r in pareto:
        print(f"  t={r['t']}  MRR={r['mrr']:.4f}  lat={r['lat']:.2f} ms")

    strict = [r for r in rows if r["mrr"] > CHAMP_MRR and r["lat"] < CHAMP_LAT]
    print()
    print(
        f"=== configs that strictly dominate champion "
        f"(MRR > {CHAMP_MRR} AND lat < {CHAMP_LAT}) ==="
    )
    print(f"count: {len(strict)}")
    strict.sort(key=lambda r: -r["mrr"])
    for r in strict[:15]:
        print(f"  t={r['t']}  MRR={r['mrr']:.4f}  lat={r['lat']:.2f} ms")

    print()
    print("=== MRR distribution under cap ===")
    buckets = [
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.3638),
        (0.3638, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
    ]
    for lo, hi in buckets:
        n = sum(1 for r in under if lo <= r["mrr"] < hi)
        print(f"  [{lo:.4f}, {hi:.4f}): {n:>5}")


if __name__ == "__main__":
    main()
