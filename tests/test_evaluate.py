"""Tests for src/evaluate.py — MRR@10 computation and save_results."""

import json

from src.evaluate import compute_mrr_at_k, save_results


def test_mrr_single_query_relevant_at_rank2():
    """1 query, 3 docs, first relevant at rank 2 → MRR = 0.5."""
    qids = ["q1", "q1", "q1"]
    scores = [0.9, 0.5, 0.7]  # sorted: 0.9(0), 0.7(1), 0.5(0) → relevant at rank 2
    labels = [0, 0, 1]
    assert compute_mrr_at_k(qids, scores, labels, k=10) == 0.5


def test_mrr_single_query_relevant_at_rank1():
    """Relevant doc has the highest score → MRR = 1.0."""
    qids = ["q1", "q1", "q1"]
    scores = [0.9, 0.5, 0.3]
    labels = [1, 0, 0]
    assert compute_mrr_at_k(qids, scores, labels, k=10) == 1.0


def test_mrr_no_relevant_docs():
    """No relevant docs → query is skipped, MRR = 0.0."""
    qids = ["q1", "q1", "q1"]
    scores = [0.9, 0.7, 0.5]
    labels = [0, 0, 0]
    assert compute_mrr_at_k(qids, scores, labels, k=10) == 0.0


def test_mrr_skips_queries_without_positives():
    """q1 has no positives (skipped), q2 relevant at rank 1 → MRR = 1.0."""
    qids = ["q1", "q1", "q2", "q2"]
    scores = [0.9, 0.3, 0.8, 0.6]
    labels = [0, 0, 1, 0]
    assert compute_mrr_at_k(qids, scores, labels, k=10) == 1.0


def test_mrr_multiple_queries():
    """Two queries: q1 relevant at rank 1 (RR=1.0), q2 relevant at rank 2 (RR=0.5).
    Mean = 0.75."""
    qids = ["q1", "q1", "q2", "q2"]
    scores = [0.9, 0.3, 0.8, 0.6]
    labels = [1, 0, 0, 1]
    mrr = compute_mrr_at_k(qids, scores, labels, k=10)
    assert abs(mrr - 0.75) < 1e-9


def test_mrr_k_cutoff():
    """Relevant doc at rank 3 with k=2 → should contribute 0."""
    qids = ["q1", "q1", "q1"]
    scores = [0.9, 0.7, 0.5]
    labels = [0, 0, 1]  # relevant at rank 3
    assert compute_mrr_at_k(qids, scores, labels, k=2) == 0.0


def test_mrr_k_cutoff_boundary():
    """Relevant doc at rank 2 with k=2 → should contribute 0.5."""
    qids = ["q1", "q1", "q1"]
    scores = [0.9, 0.7, 0.5]
    labels = [0, 1, 0]
    assert compute_mrr_at_k(qids, scores, labels, k=2) == 0.5


def test_mrr_empty_input():
    """Empty input → MRR = 0.0."""
    assert compute_mrr_at_k([], [], [], k=10) == 0.0


# ---------------------------------------------------------------------------
# Edge cases for compute_mrr_at_k
# ---------------------------------------------------------------------------

def test_mrr_tied_scores():
    """Two docs with the same score, one relevant. MRR should be > 0."""
    qids = ["q1", "q1"]
    scores = [0.5, 0.5]
    labels = [0, 1]
    mrr = compute_mrr_at_k(qids, scores, labels, k=10)
    assert mrr > 0, "Tied scores with one relevant doc should yield MRR > 0"
    assert mrr <= 1.0


def test_mrr_all_relevant():
    """All docs relevant → first relevant at rank 1 → MRR = 1.0."""
    qids = ["q1", "q1", "q1"]
    scores = [0.9, 0.7, 0.5]
    labels = [1, 1, 1]
    assert compute_mrr_at_k(qids, scores, labels, k=10) == 1.0


def test_mrr_many_queries():
    """100 queries each with 10 docs, result should be in (0, 1]."""
    qids = []
    scores = []
    labels = []
    for q in range(100):
        for d in range(10):
            qids.append(f"q{q}")
            scores.append(float(10 - d))  # rank 1 has highest score
            labels.append(1 if d == q % 10 else 0)  # relevant doc at position q%10
    mrr = compute_mrr_at_k(qids, scores, labels, k=10)
    assert 0.0 < mrr <= 1.0


def test_mrr_single_doc_relevant():
    """One query with one doc that's relevant → MRR = 1.0."""
    assert compute_mrr_at_k(["q1"], [0.5], [1], k=10) == 1.0


def test_mrr_single_doc_not_relevant():
    """One query with one doc not relevant → query skipped → MRR = 0.0."""
    assert compute_mrr_at_k(["q1"], [0.5], [0], k=10) == 0.0


# ---------------------------------------------------------------------------
# save_results tests
# ---------------------------------------------------------------------------

def test_save_results_creates_file(tmp_path):
    path = str(tmp_path / "results.json")
    save_results({"key": "value"}, path)
    assert (tmp_path / "results.json").exists()


def test_save_results_valid_json(tmp_path):
    path = str(tmp_path / "results.json")
    data = {"mrr10": 0.73, "latency_ms": 36.5, "system": "baseline_a"}
    save_results(data, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data


def test_save_results_overwrites(tmp_path):
    path = str(tmp_path / "results.json")
    save_results({"version": 1}, path)
    save_results({"version": 2}, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["version"] == 2


def test_save_results_nested_dict(tmp_path):
    path = str(tmp_path / "results.json")
    data = {
        "system": "baseline_b",
        "thresholds": [0.01, 0.05, 0.1],
        "exit_counts": {"layer_0": 10, "layer_5": 54},
    }
    save_results(data, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded == data
