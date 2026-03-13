"""Tests for Baseline A inference (src/inference.py).

These tests require the pre-tokenized dev set and a GPU.
They use a small batch to keep runtime short.
"""

import os

import pytest
import torch


# Skip all tests if no CUDA or no tokenized data
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(
        not os.path.exists("data/dev_tokenized.pt"),
        reason="Pre-tokenized dev set not found",
    ),
]


@pytest.fixture(scope="module")
def baseline_a_results():
    """Run Baseline A once with a small batch for all tests in this module."""
    from src.inference import run_baseline_a

    return run_baseline_a(
        tokenized_path="data/dev_tokenized.pt",
        batch_size=8,
        output_dir="results",
    )


def test_result_keys(baseline_a_results):
    """Result dict has all expected keys."""
    expected_keys = {"system", "mrr10", "mean_batch_latency_ms", "total_latency_s", "batch_size"}
    assert expected_keys == set(baseline_a_results.keys())


def test_mrr_in_range(baseline_a_results):
    """MRR@10 should be in (0, 1]."""
    mrr = baseline_a_results["mrr10"]
    assert 0.0 < mrr <= 1.0, f"MRR@10 out of range: {mrr}"


def test_latency_positive(baseline_a_results):
    """Latency values should be positive."""
    assert baseline_a_results["mean_batch_latency_ms"] > 0
    assert baseline_a_results["total_latency_s"] > 0


def test_batch_size_recorded(baseline_a_results):
    """Batch size in results should match what we passed."""
    assert baseline_a_results["batch_size"] == 8


def test_system_name(baseline_a_results):
    """System name should be baseline_a."""
    assert baseline_a_results["system"] == "baseline_a"
