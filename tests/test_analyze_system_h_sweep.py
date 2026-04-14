"""Tests for ``scripts/analyze_system_h_sweep.py``.

The System H analyzer loads per-patience sweep CSVs from
``results/system_h_sweep_results/system_h_p<P>_t0=*_t1=*.csv`` and reports
ranking, Pareto frontier, and dominance counts for a single patience band
at a time (selected via ``--patience``).

These tests exercise the pure helper functions in isolation on tiny
synthetic CSVs so the analyzer logic is verified without touching the
real sweep outputs.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_system_h_sweep.py"


@pytest.fixture(scope="module")
def analyzer():
    """Load ``scripts/analyze_system_h_sweep.py`` as a module."""
    spec = importlib.util.spec_from_file_location("analyze_system_h_sweep", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_system_h_sweep"] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "t0", "t1", "t2", "t3", "t4",
        "patience",
        "mrr10", "mean_batch_latency_ms",
        "exit_count_0", "exit_count_1", "exit_count_2",
        "exit_count_3", "exit_count_4", "exit_count_5",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _row(
    t0=0.001, t1=0.001, t2=0.001, t3=0.001, t4=0.001,
    patience=2, mrr=0.5, lat=16.0,
    ec=(100, 0, 0, 0, 0, 0),
):
    return {
        "t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4,
        "patience": patience,
        "mrr10": mrr, "mean_batch_latency_ms": lat,
        "exit_count_0": ec[0], "exit_count_1": ec[1], "exit_count_2": ec[2],
        "exit_count_3": ec[3], "exit_count_4": ec[4], "exit_count_5": ec[5],
    }


class TestLoadRows:
    def test_filters_by_patience_via_filename_prefix(self, analyzer, tmp_path):
        p2_csv = tmp_path / "system_h_p2_t0=0.001_t1=0.001.csv"
        p3_csv = tmp_path / "system_h_p3_t0=0.001_t1=0.001.csv"
        _write_csv(p2_csv, [_row(patience=2, mrr=0.42)])
        _write_csv(p3_csv, [_row(patience=3, mrr=0.47)])

        rows_p2 = analyzer.load_rows(str(tmp_path), patience=2)
        rows_p3 = analyzer.load_rows(str(tmp_path), patience=3)

        assert len(rows_p2) == 1 and rows_p2[0]["mrr"] == pytest.approx(0.42)
        assert len(rows_p3) == 1 and rows_p3[0]["mrr"] == pytest.approx(0.47)

    def test_parses_all_fields_with_correct_types(self, analyzer, tmp_path):
        csv_path = tmp_path / "system_h_p4_t0=0.001_t1=0.003.csv"
        _write_csv(csv_path, [_row(
            t0=0.001, t1=0.003, t2=0.01, t3=0.03, t4=0.1,
            patience=4, mrr=0.5580, lat=16.48,
            ec=(10, 20, 30, 40, 50, 60),
        )])

        rows = analyzer.load_rows(str(tmp_path), patience=4)

        assert len(rows) == 1
        r = rows[0]
        assert r["t"] == (0.001, 0.003, 0.01, 0.03, 0.1)
        assert r["mrr"] == pytest.approx(0.5580)
        assert r["lat"] == pytest.approx(16.48)
        assert r["ec"] == [10, 20, 30, 40, 50, 60]
        assert all(isinstance(c, int) for c in r["ec"])

    def test_returns_empty_when_no_matching_files(self, analyzer, tmp_path):
        # Different-patience file present; asking for P=5 yields empty.
        _write_csv(
            tmp_path / "system_h_p2_t0=0.001_t1=0.001.csv",
            [_row(patience=2)],
        )
        assert analyzer.load_rows(str(tmp_path), patience=5) == []

    def test_loads_multiple_files_and_concats_rows(self, analyzer, tmp_path):
        _write_csv(
            tmp_path / "system_h_p3_t0=0.001_t1=0.001.csv",
            [_row(patience=3, t1=0.001, mrr=0.30),
             _row(patience=3, t1=0.001, mrr=0.31)],
        )
        _write_csv(
            tmp_path / "system_h_p3_t0=0.001_t1=0.003.csv",
            [_row(patience=3, t1=0.003, mrr=0.32)],
        )
        rows = analyzer.load_rows(str(tmp_path), patience=3)
        assert len(rows) == 3
        assert sorted(r["mrr"] for r in rows) == pytest.approx([0.30, 0.31, 0.32])


class TestConfigsUnderCap:
    def test_filters_by_latency_and_sorts_by_mrr_desc(self, analyzer):
        rows = [
            {"t": (0,), "mrr": 0.4, "lat": 16.0, "ec": [0] * 6},
            {"t": (1,), "mrr": 0.6, "lat": 17.0, "ec": [0] * 6},  # over cap
            {"t": (2,), "mrr": 0.5, "lat": 15.0, "ec": [0] * 6},
            {"t": (3,), "mrr": 0.45, "lat": 16.5, "ec": [0] * 6},
        ]
        out = analyzer.configs_under_cap(rows, cap=16.5)
        assert [r["mrr"] for r in out] == [0.5, 0.45, 0.4]

    def test_cap_is_inclusive(self, analyzer):
        rows = [{"t": (0,), "mrr": 0.5, "lat": 16.5, "ec": [0] * 6}]
        assert analyzer.configs_under_cap(rows, cap=16.5) == rows


class TestParetoFrontier:
    def test_single_point_is_its_own_frontier(self, analyzer):
        rows = [{"t": (0,), "mrr": 0.5, "lat": 16.0, "ec": [0] * 6}]
        assert analyzer.pareto_frontier(rows) == rows

    def test_strictly_dominated_points_excluded(self, analyzer):
        rows = [
            {"t": (0,), "mrr": 0.3, "lat": 15.0, "ec": [0] * 6},  # on frontier
            {"t": (1,), "mrr": 0.4, "lat": 16.0, "ec": [0] * 6},  # on frontier
            {"t": (2,), "mrr": 0.35, "lat": 16.5, "ec": [0] * 6},  # dominated by (1,)
            {"t": (3,), "mrr": 0.5, "lat": 17.0, "ec": [0] * 6},  # on frontier
        ]
        front = analyzer.pareto_frontier(rows)
        mrrs = [r["mrr"] for r in front]
        lats = [r["lat"] for r in front]
        assert mrrs == [0.3, 0.4, 0.5]
        assert lats == [15.0, 16.0, 17.0]


class TestStrictDominators:
    def test_requires_both_strict_inequalities(self, analyzer):
        rows = [
            {"t": (0,), "mrr": 0.50, "lat": 15.0, "ec": [0] * 6},  # dominates
            {"t": (1,), "mrr": 0.40, "lat": 16.0, "ec": [0] * 6},  # tied on mrr → NO
            {"t": (2,), "mrr": 0.45, "lat": 16.0, "ec": [0] * 6},  # tied on lat → NO
            {"t": (3,), "mrr": 0.39, "lat": 15.9, "ec": [0] * 6},  # worse mrr → NO
            {"t": (4,), "mrr": 0.60, "lat": 14.0, "ec": [0] * 6},  # dominates
        ]
        out = analyzer.strict_dominators(rows, ref_mrr=0.40, ref_lat=16.0)
        assert {r["mrr"] for r in out} == {0.50, 0.60}
