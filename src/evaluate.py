import json
from collections import defaultdict
from typing import List


def compute_mrr_at_k(
    qids: List[str], scores: List[float], labels: List[int], k: int = 10
) -> float:
    """Compute Mean Reciprocal Rank at k.

    Groups documents by query ID, sorts by descending score,
    and finds the rank of the first relevant document.
    """
    # Group by qid
    query_docs = defaultdict(list)
    for qid, score, label in zip(qids, scores, labels):
        query_docs[qid].append((score, label))

    reciprocal_ranks = []
    for qid, docs in query_docs.items():
        # Skip queries with no relevant documents (MS MARCO standard)
        if not any(label == 1 for _, label in docs):
            continue
        # Sort by descending score
        docs_sorted = sorted(docs, key=lambda x: x[0], reverse=True)
        rr = 0.0
        for rank, (_, label) in enumerate(docs_sorted, start=1):
            if rank > k:
                break
            if label == 1:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def save_results(results_dict: dict, path: str) -> None:
    """Save a results dictionary to JSON."""
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
