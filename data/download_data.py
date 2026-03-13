"""
Download MS MARCO passage ranking data and produce fixed train/dev splits.

Train: 10K queries, ~1M pairs (top-100 BM25 candidates per query)
Dev:   1K queries, 100K pairs (top-100 candidates per query)

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


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_via_ir_datasets():
    """Try loading MS MARCO via ir_datasets (preferred — gives qrels + top-1000)."""
    import ir_datasets

    # --- Training split ---
    print("Loading MS MARCO training data via ir_datasets ...")
    train_ds = ir_datasets.load("msmarco-passage/train/judged")

    # Build lookup tables
    docs_store = train_ds.docs_store()
    queries = {q.query_id: q.text for q in train_ds.queries_iter()}
    # Collect qrels (relevance judgments)
    qrels = {}
    for qrel in train_ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    # Sample TRAIN_QUERIES queries deterministically from queries that have qrels
    all_train_qids = sorted(qrels.keys())
    random.seed(SEED)
    sampled_train_qids = set(random.sample(all_train_qids, min(TRAIN_QUERIES, len(all_train_qids))))

    # Stream scored docs but only keep entries for sampled queries
    query_docs = {}
    for sd in tqdm(train_ds.scoreddocs_iter(), desc="Reading train scored docs"):
        if sd.query_id in sampled_train_qids:
            query_docs.setdefault(sd.query_id, []).append(sd.doc_id)

    train_rows = []
    for qid in tqdm(sorted(sampled_train_qids), desc="Building train split"):
        doc_ids = query_docs.get(qid, [])[:CANDIDATES_PER_QUERY]
        query_text = queries[qid]
        for did in doc_ids:
            doc = docs_store.get(did)
            label = 1 if qrels.get(qid, {}).get(did, 0) > 0 else 0
            train_rows.append({
                "qid": str(qid),
                "docid": str(did),
                "query": query_text,
                "passage": doc.text,
                "label": label,
            })

    # --- Dev split ---
    print("Loading MS MARCO dev data via ir_datasets ...")
    dev_ds = ir_datasets.load("msmarco-passage/dev/small")

    dev_docs_store = dev_ds.docs_store()
    dev_queries = {q.query_id: q.text for q in dev_ds.queries_iter()}
    dev_qrels = {}
    for qrel in dev_ds.qrels_iter():
        dev_qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    dev_query_docs = {}
    for sd in tqdm(dev_ds.scoreddocs_iter(), desc="Reading dev scored docs"):
        dev_query_docs.setdefault(sd.query_id, []).append(sd.doc_id)

    # First DEV_QUERIES queries in sorted qid order
    all_dev_qids = sorted(dev_query_docs.keys())[:DEV_QUERIES]

    dev_rows = []
    for qid in tqdm(all_dev_qids, desc="Building dev split"):
        doc_ids = dev_query_docs[qid][:CANDIDATES_PER_QUERY]
        query_text = dev_queries[qid]
        for did in doc_ids:
            doc = dev_docs_store.get(did)
            label = 1 if dev_qrels.get(qid, {}).get(did, 0) > 0 else 0
            dev_rows.append({
                "qid": str(qid),
                "docid": str(did),
                "query": query_text,
                "passage": doc.text,
                "label": label,
            })

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

    # Tokenize in batches to avoid excessive memory usage
    batch_size = 10_000
    all_input_ids = []
    all_attention_mask = []

    for start in tqdm(range(0, len(queries), batch_size), desc="Tokenizing"):
        end = min(start + batch_size, len(queries))
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

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_mask, dim=0)

    tokenized = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
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

    # Build DataFrames
    train_df = pd.DataFrame(train_rows)
    dev_df = pd.DataFrame(dev_rows)

    # Print statistics
    print(f"\nTraining split: {train_df['qid'].nunique()} queries, "
          f"{len(train_df)} pairs, "
          f"{train_df['label'].sum()} positives")
    print(f"Dev split:      {dev_df['qid'].nunique()} queries, "
          f"{len(dev_df)} pairs, "
          f"{dev_df['label'].sum()} positives")

    # Save Parquet files
    train_path = DATA_DIR / "msmarco_train.parquet"
    dev_path = DATA_DIR / "msmarco_dev.parquet"
    train_df.to_parquet(train_path, index=False)
    dev_df.to_parquet(dev_path, index=False)
    print(f"\nSaved to {train_path}")
    print(f"Saved to {dev_path}")

    # Pre-tokenize dev split
    tokenized_path = DATA_DIR / "dev_tokenized.pt"
    pre_tokenize_dev(dev_df, tokenized_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
