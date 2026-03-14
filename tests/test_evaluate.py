"""Tests for src/evaluate.py — MRR@10 computation."""

from src.evaluate import compute_mrr_at_k


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
