"""Split the training set into train/val by query ID.

Carves out a validation set from msmarco_train.parquet for hyperparameter
tuning. The full 744-query dev set (dev_tokenized.pt) remains untouched
as the final test set.

Output files:
  - msmarco_train_split.parquet  (training queries, ~95%)
  - msmarco_val.parquet          (validation queries, ~5%)
  - val_tokenized.pt             (pre-tokenized val for inference)
"""

import argparse
import random
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.constants import MODEL_NAME, MAX_TOKEN_LENGTH

TOKENIZE_BATCH_SIZE = 10_000


def split_train_val(
    input_path: str = "data/msmarco_train.parquet",
    output_dir: str = "data",
    val_fraction: float = 0.05,
    seed: int = 42,
):
    """Split training data into train/val by query ID.

    Args:
        input_path: Path to the full training parquet.
        output_dir: Directory to write split files.
        val_fraction: Fraction of queries to hold out for validation.
        seed: Random seed for deterministic splitting.
    """
    df = pd.read_parquet(input_path)
    all_qids = sorted(df["qid"].unique())

    random.seed(seed)
    num_val = max(1, int(len(all_qids) * val_fraction))
    val_qids = set(random.sample(all_qids, num_val))

    val_mask = df["qid"].isin(val_qids)
    train_df = df[~val_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "msmarco_train_split.parquet"
    val_path = output_dir / "msmarco_val.parquet"
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Train split: {train_df['qid'].nunique()} queries, {len(train_df)} pairs")
    print(f"Val split:   {val_df['qid'].nunique()} queries, {len(val_df)} pairs")
    print(f"Saved to {train_path}")
    print(f"Saved to {val_path}")

    # Pre-tokenize val set
    tokenized_path = output_dir / "val_tokenized.pt"
    _pre_tokenize(val_df, tokenized_path)

    return train_path, val_path, tokenized_path


def _pre_tokenize(df: pd.DataFrame, output_path: Path):
    """Tokenize query-passage pairs and save tensors to disk."""
    print(f"Pre-tokenizing {len(df)} val pairs ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    queries = df["query"].tolist()
    passages = df["passage"].tolist()

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    for start in tqdm(range(0, len(queries), TOKENIZE_BATCH_SIZE), desc="Tokenizing val"):
        end = min(start + TOKENIZE_BATCH_SIZE, len(queries))
        encoded = tokenizer(
            queries[start:end],
            passages[start:end],
            max_length=MAX_TOKEN_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        all_input_ids.append(encoded["input_ids"])
        all_attention_mask.append(encoded["attention_mask"])
        all_token_type_ids.append(encoded["token_type_ids"])

    tokenized = {
        "input_ids": torch.cat(all_input_ids, dim=0),
        "attention_mask": torch.cat(all_attention_mask, dim=0),
        "token_type_ids": torch.cat(all_token_type_ids, dim=0),
        "qids": df["qid"].tolist(),
        "labels": df["label"].tolist(),
    }

    torch.save(tokenized, output_path)
    print(f"Saved pre-tokenized val set to {output_path}")
    print(f"  shape: {tokenized['input_ids'].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="data/msmarco_train.parquet")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_train_val(
        input_path=args.input_path,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
