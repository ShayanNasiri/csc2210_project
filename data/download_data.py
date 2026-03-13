"""
Download MS MARCO passage ranking data and produce fixed train/dev splits.

Train: ~10K queries, ~1M pairs (top-100 BM25 candidates per query)
Dev:   1K queries, ~97K pairs (top-100 candidates per query)

Also pre-tokenizes the dev split to data/dev_tokenized.pt for latency-free
GPU inference measurement.
"""

import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent
SEED = 42
TRAIN_QUERIES = 10_000
DEV_QUERIES = 1_000
CANDIDATES_PER_QUERY = 100
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512
TOKENIZE_BATCH_SIZE = 10_000


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_split(dataset, query_ids, docs_store, queries, qrels, desc):
    """Build rows for a set of query IDs from an ir_datasets split."""
    rows = []
    for qid in tqdm(sorted(query_ids), desc=desc):
        doc_ids = query_ids[qid][:CANDIDATES_PER_QUERY]
        query_text = queries[qid]
        for did in doc_ids:
            doc = docs_store.get(did)
            label = 1 if qrels.get(qid, {}).get(did, 0) > 0 else 0
            rows.append({
                "qid": str(qid),
                "docid": str(did),
                "query": query_text,
                "passage": doc.text,
                "label": label,
            })
    return rows


def _collect_qrels(dataset):
    """Collect relevance judgments into a dict of dicts."""
    qrels = {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
    return qrels


def _collect_scored_docs(dataset, keep_qids=None, desc="Reading scored docs"):
    """Stream scored docs, optionally filtering to a set of query IDs."""
    query_docs = {}
    for sd in tqdm(dataset.scoreddocs_iter(), desc=desc):
        if keep_qids is None or sd.query_id in keep_qids:
            query_docs.setdefault(sd.query_id, []).append(sd.doc_id)
    return query_docs


def load_via_ir_datasets():
    """Load MS MARCO train/dev via ir_datasets."""
    import ir_datasets

    # --- Training split ---
    print("Loading MS MARCO training data via ir_datasets ...")
    train_ds = ir_datasets.load("msmarco-passage/train/judged")

    docs_store = train_ds.docs_store()
    queries = {q.query_id: q.text for q in train_ds.queries_iter()}
    qrels = _collect_qrels(train_ds)

    # Sample queries deterministically from those with qrels
    random.seed(SEED)
    sampled_qids = set(random.sample(
        sorted(qrels.keys()),
        min(TRAIN_QUERIES, len(qrels)),
    ))

    query_docs = _collect_scored_docs(
        train_ds, keep_qids=sampled_qids, desc="Reading train scored docs",
    )
    train_rows = _load_split(
        train_ds, query_docs, docs_store, queries, qrels, "Building train split",
    )

    # --- Dev split ---
    print("Loading MS MARCO dev data via ir_datasets ...")
    dev_ds = ir_datasets.load("msmarco-passage/dev/small")

    dev_docs_store = dev_ds.docs_store()
    dev_queries = {q.query_id: q.text for q in dev_ds.queries_iter()}
    dev_qrels = _collect_qrels(dev_ds)

    all_dev_docs = _collect_scored_docs(dev_ds, desc="Reading dev scored docs")
    # First DEV_QUERIES queries in sorted qid order
    selected_dev_docs = {
        qid: all_dev_docs[qid]
        for qid in sorted(all_dev_docs.keys())[:DEV_QUERIES]
    }
    dev_rows = _load_split(
        dev_ds, selected_dev_docs, dev_docs_store, dev_queries, dev_qrels,
        "Building dev split",
    )

    return train_rows, dev_rows


# ---------------------------------------------------------------------------
# Pre-tokenize dev split
# ---------------------------------------------------------------------------

def pre_tokenize_dev(dev_df: pd.DataFrame, output_path: Path):
    """Tokenize all dev query-passage pairs and save tensors to disk."""
    print(f"Pre-tokenizing {len(dev_df)} dev pairs ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    queries = dev_df["query"].tolist()
    passages = dev_df["passage"].tolist()

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    for start in tqdm(range(0, len(queries), TOKENIZE_BATCH_SIZE), desc="Tokenizing"):
        end = min(start + TOKENIZE_BATCH_SIZE, len(queries))
        encoded = tokenizer(
            queries[start:end],
            passages[start:end],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_mask.append(encoded["attention_mask"])
        all_token_type_ids.append(encoded["token_type_ids"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)
    token_type_ids = torch.cat(all_token_type_ids, dim=0)

    tokenized = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "qids": dev_df["qid"].tolist(),
        "labels": dev_df["label"].tolist(),
    }

    torch.save(tokenized, output_path)
    print(f"Saved pre-tokenized dev set to {output_path}")
    print(f"  input_ids shape:      {input_ids.shape}")
    print(f"  attention_mask shape: {attention_mask.shape}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    train_rows, dev_rows = load_via_ir_datasets()

    train_df = pd.DataFrame(train_rows)
    dev_df = pd.DataFrame(dev_rows)

    print(f"\nTraining split: {train_df['qid'].nunique()} queries, "
          f"{len(train_df)} pairs, "
          f"{train_df['label'].sum()} positives")
    print(f"Dev split:      {dev_df['qid'].nunique()} queries, "
          f"{len(dev_df)} pairs, "
          f"{dev_df['label'].sum()} positives")

    train_path = DATA_DIR / "msmarco_train.parquet"
    dev_path = DATA_DIR / "msmarco_dev.parquet"
    train_df.to_parquet(train_path, index=False)
    dev_df.to_parquet(dev_path, index=False)
    print(f"\nSaved to {train_path}")
    print(f"Saved to {dev_path}")

    tokenized_path = DATA_DIR / "dev_tokenized.pt"
    pre_tokenize_dev(dev_df, tokenized_path)

    print("\nDone.")


if __name__ == "__main__":
    import sys

    if "--retokenize" in sys.argv:
        # Re-tokenize dev set only (no download)
        dev_df = pd.read_parquet(DATA_DIR / "msmarco_dev.parquet")
        pre_tokenize_dev(dev_df, DATA_DIR / "dev_tokenized.pt")
    else:
        main()
