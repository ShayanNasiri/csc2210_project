"""Tests for train/val split — carving a validation set from the training data.

Split: 95% train / 5% val by query ID, deterministic (seeded).
Val is pre-tokenized to val_tokenized.pt for inference evaluation.
The full 744-query dev set remains untouched as the test set.
"""

import os

import pandas as pd
import pytest
import torch

from src.constants import MODEL_NAME, MAX_TOKEN_LENGTH


# ---- Constants ----

def test_val_data_path_constant():
    """Constants must define a path for the val tokenized data."""
    from src.constants import DEFAULT_VAL_DATA_PATH
    assert "val" in DEFAULT_VAL_DATA_PATH.lower()
    from src.constants import DEFAULT_DEV_DATA_PATH
    assert DEFAULT_VAL_DATA_PATH != DEFAULT_DEV_DATA_PATH


def test_train_split_path_constant():
    """Constants must define a path for the split training parquet."""
    from src.constants import DEFAULT_TRAIN_SPLIT_PATH
    assert "train" in DEFAULT_TRAIN_SPLIT_PATH.lower()


# ---- split_train_val function ----

def test_split_train_val_importable():
    from data.split_train_val import split_train_val


def test_split_train_val_signature():
    import inspect
    from data.split_train_val import split_train_val
    sig = inspect.signature(split_train_val)
    params = set(sig.parameters.keys())
    assert "input_path" in params
    assert "output_dir" in params
    assert "val_fraction" in params
    assert "seed" in params


# ---- Split logic tests ----

@pytest.fixture
def sample_train_parquet(tmp_path):
    """Create a parquet with 20 queries, 4 docs each = 80 rows."""
    rows = []
    for q in range(20):
        for d in range(4):
            rows.append({
                "qid": f"q{q}",
                "docid": f"d{q}_{d}",
                "query": f"query {q}",
                "passage": f"passage {q} {d}",
                "label": 1 if d == 0 else 0,
            })
    df = pd.DataFrame(rows)
    path = str(tmp_path / "train.parquet")
    df.to_parquet(path, index=False)
    return path


def test_split_produces_two_parquets(sample_train_parquet, tmp_path):
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    assert os.path.exists(tmp_path / "msmarco_train_split.parquet")
    assert os.path.exists(tmp_path / "msmarco_val.parquet")


def test_split_produces_tokenized_val(sample_train_parquet, tmp_path):
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    assert os.path.exists(tmp_path / "val_tokenized.pt")


def test_split_no_query_overlap(sample_train_parquet, tmp_path):
    """Train and val must have disjoint query IDs."""
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    train_df = pd.read_parquet(tmp_path / "msmarco_train_split.parquet")
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    train_qids = set(train_df["qid"].unique())
    val_qids = set(val_df["qid"].unique())
    assert train_qids.isdisjoint(val_qids), "Train and val must have no shared queries"


def test_split_preserves_all_rows(sample_train_parquet, tmp_path):
    """Total rows in train + val must equal original."""
    from data.split_train_val import split_train_val
    original_df = pd.read_parquet(sample_train_parquet)
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    train_df = pd.read_parquet(tmp_path / "msmarco_train_split.parquet")
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    assert len(train_df) + len(val_df) == len(original_df)


def test_split_val_fraction_approximate(sample_train_parquet, tmp_path):
    """Val should contain approximately val_fraction of the queries."""
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    train_df = pd.read_parquet(tmp_path / "msmarco_train_split.parquet")
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    total_queries = train_df["qid"].nunique() + val_df["qid"].nunique()
    val_ratio = val_df["qid"].nunique() / total_queries
    # With 20 queries and 25% split, expect 5 val queries (exact)
    assert 0.15 <= val_ratio <= 0.35


def test_split_deterministic(sample_train_parquet, tmp_path):
    """Same seed must produce identical splits."""
    from data.split_train_val import split_train_val
    dir1 = tmp_path / "run1"
    dir2 = tmp_path / "run2"
    dir1.mkdir()
    dir2.mkdir()
    split_train_val(input_path=sample_train_parquet, output_dir=str(dir1), seed=42)
    split_train_val(input_path=sample_train_parquet, output_dir=str(dir2), seed=42)
    val1 = pd.read_parquet(dir1 / "msmarco_val.parquet")
    val2 = pd.read_parquet(dir2 / "msmarco_val.parquet")
    assert set(val1["qid"].unique()) == set(val2["qid"].unique())


def test_split_different_seeds_differ(sample_train_parquet, tmp_path):
    """Different seeds should produce different splits."""
    from data.split_train_val import split_train_val
    dir1 = tmp_path / "seed1"
    dir2 = tmp_path / "seed2"
    dir1.mkdir()
    dir2.mkdir()
    split_train_val(input_path=sample_train_parquet, output_dir=str(dir1), seed=42)
    split_train_val(input_path=sample_train_parquet, output_dir=str(dir2), seed=99)
    val1 = pd.read_parquet(dir1 / "msmarco_val.parquet")
    val2 = pd.read_parquet(dir2 / "msmarco_val.parquet")
    # With 20 queries, different seeds should (very likely) pick different val sets
    assert set(val1["qid"].unique()) != set(val2["qid"].unique())


def test_split_columns_preserved(sample_train_parquet, tmp_path):
    """Both splits must have the same columns as the original."""
    from data.split_train_val import split_train_val
    original_df = pd.read_parquet(sample_train_parquet)
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    train_df = pd.read_parquet(tmp_path / "msmarco_train_split.parquet")
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    assert list(train_df.columns) == list(original_df.columns)
    assert list(val_df.columns) == list(original_df.columns)


def test_split_val_has_positives(sample_train_parquet, tmp_path):
    """Val set must contain at least some positive labels for MRR evaluation."""
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    assert val_df["label"].sum() > 0


# ---- Tokenized val tests ----

def test_tokenized_val_keys(sample_train_parquet, tmp_path):
    """Tokenized val must have the same keys as dev_tokenized.pt."""
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    tokenized = torch.load(tmp_path / "val_tokenized.pt", weights_only=False)
    expected_keys = {"input_ids", "attention_mask", "token_type_ids", "qids", "labels"}
    assert set(tokenized.keys()) == expected_keys


def test_tokenized_val_shapes(sample_train_parquet, tmp_path):
    """Tokenized val tensors must have correct shapes."""
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    tokenized = torch.load(tmp_path / "val_tokenized.pt", weights_only=False)
    n = len(val_df)
    assert tokenized["input_ids"].shape == (n, MAX_TOKEN_LENGTH)
    assert tokenized["attention_mask"].shape == (n, MAX_TOKEN_LENGTH)
    assert tokenized["token_type_ids"].shape == (n, MAX_TOKEN_LENGTH)
    assert len(tokenized["qids"]) == n
    assert len(tokenized["labels"]) == n


def test_tokenized_val_rows_match_parquet(sample_train_parquet, tmp_path):
    """Tokenized val must have the same number of rows as the val parquet."""
    from data.split_train_val import split_train_val
    split_train_val(
        input_path=sample_train_parquet,
        output_dir=str(tmp_path),
        val_fraction=0.25,
    )
    val_df = pd.read_parquet(tmp_path / "msmarco_val.parquet")
    tokenized = torch.load(tmp_path / "val_tokenized.pt", weights_only=False)
    assert tokenized["input_ids"].shape[0] == len(val_df)
