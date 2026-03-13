"""Validate Phase 1 data files: parquet splits and pre-tokenized dev set."""

from pathlib import Path

import pandas as pd
import pytest
import torch

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture
def train_df():
    path = DATA_DIR / "msmarco_train.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture
def dev_df():
    path = DATA_DIR / "msmarco_dev.parquet"
    assert path.exists(), f"Missing {path}"
    return pd.read_parquet(path)


@pytest.fixture
def dev_tokenized():
    path = DATA_DIR / "dev_tokenized.pt"
    assert path.exists(), f"Missing {path}"
    return torch.load(path, weights_only=False)


# --- Train split ---

def test_train_columns(train_df):
    assert list(train_df.columns) == ["qid", "docid", "query", "passage", "label"]


def test_train_query_count(train_df):
    n = train_df["qid"].nunique()
    assert n > 9900, f"Expected ~10K queries, got {n}"


def test_train_row_count(train_df):
    assert len(train_df) > 900000, f"Expected ~1M rows, got {len(train_df)}"


def test_train_labels(train_df):
    assert set(train_df["label"].unique()).issubset({0, 1})


def test_train_no_nulls(train_df):
    assert train_df.notna().all().all(), "Train has null values"


# --- Dev split ---

def test_dev_columns(dev_df):
    assert list(dev_df.columns) == ["qid", "docid", "query", "passage", "label"]


def test_dev_query_count(dev_df):
    assert dev_df["qid"].nunique() == 1000


def test_dev_row_count(dev_df):
    assert len(dev_df) > 90000, f"Expected ~97K rows, got {len(dev_df)}"


def test_dev_labels(dev_df):
    assert set(dev_df["label"].unique()).issubset({0, 1})


def test_dev_no_nulls(dev_df):
    assert dev_df.notna().all().all(), "Dev has null values"


# --- Pre-tokenized dev ---

def test_tokenized_keys(dev_tokenized):
    assert set(dev_tokenized.keys()) == {"input_ids", "attention_mask", "qids", "labels"}


def test_tokenized_shape(dev_tokenized):
    assert dev_tokenized["input_ids"].shape[1] == 512
    assert dev_tokenized["attention_mask"].shape[1] == 512


def test_tokenized_rows_match_dev(dev_df, dev_tokenized):
    assert dev_tokenized["input_ids"].shape[0] == len(dev_df)


def test_tokenized_qids_match_dev(dev_df, dev_tokenized):
    assert len(dev_tokenized["qids"]) == len(dev_df)


def test_tokenized_labels_match_dev(dev_df, dev_tokenized):
    assert len(dev_tokenized["labels"]) == len(dev_df)
