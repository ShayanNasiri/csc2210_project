"""Tests for src/constants.py — verify all constants are correct."""

import pytest

from src.constants import (
    MODEL_NAME,
    NUM_BERT_LAYERS,
    NUM_OFFRAMPS,
    HIDDEN_SIZE,
    MAX_TOKEN_LENGTH,
    WARMUP_BATCHES,
    TIMED_BATCH_LIMIT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_DEV_DATA_PATH,
    DEFAULT_RESULTS_DIR,
    DEFAULT_OFFRAMP_WEIGHTS_PATH,
    DEFAULT_ENTROPY_THRESHOLDS,
)


def test_model_name_valid():
    assert isinstance(MODEL_NAME, str)
    assert len(MODEL_NAME) > 0


def test_num_offramps_equals_layers_minus_one():
    assert NUM_OFFRAMPS == NUM_BERT_LAYERS - 1


def test_hidden_size_matches_model():
    transformers = pytest.importorskip("transformers")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_NAME)
    assert config.hidden_size == HIDDEN_SIZE


def test_num_layers_matches_model():
    transformers = pytest.importorskip("transformers")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_NAME)
    assert config.num_hidden_layers == NUM_BERT_LAYERS


def test_max_token_length_matches_model():
    transformers = pytest.importorskip("transformers")
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(MODEL_NAME)
    assert MAX_TOKEN_LENGTH == config.max_position_embeddings


def test_entropy_thresholds_sorted():
    assert DEFAULT_ENTROPY_THRESHOLDS == sorted(DEFAULT_ENTROPY_THRESHOLDS)


def test_entropy_thresholds_positive():
    assert all(t > 0 for t in DEFAULT_ENTROPY_THRESHOLDS)


def test_batch_size_positive():
    assert DEFAULT_BATCH_SIZE > 0


def test_warmup_and_timed_positive():
    assert WARMUP_BATCHES > 0
    assert TIMED_BATCH_LIMIT > 0
