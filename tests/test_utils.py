"""Tests for src/utils.py — set_seed, get_device, TimerContext."""

import random
import time

import numpy as np
import pytest
import torch

from src.utils import TimerContext, get_device, set_seed


# ── set_seed tests ──────────────────────────────────────────────────────


def test_set_seed_python_random_reproducible():
    set_seed(42)
    a = random.random()
    set_seed(42)
    b = random.random()
    assert a == b


def test_set_seed_numpy_reproducible():
    set_seed(42)
    a = np.random.rand()
    set_seed(42)
    b = np.random.rand()
    assert a == b


def test_set_seed_torch_reproducible():
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(42)
    a = torch.randn(5)
    set_seed(99)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_set_seed_default_seed():
    set_seed()
    a = torch.randn(5)
    set_seed()
    b = torch.randn(5)
    assert torch.equal(a, b)


# ── get_device tests ───────────────────────────────────────────────────


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)


def test_get_device_type_valid():
    device = get_device()
    assert device.type in ("cuda", "cpu")


# ── TimerContext tests ─────────────────────────────────────────────────


def test_timer_cpu_elapsed_positive():
    with TimerContext(device=torch.device("cpu")) as t:
        time.sleep(0.01)
    assert t.elapsed_ms > 0


def test_timer_cpu_elapsed_reasonable():
    with TimerContext(device=torch.device("cpu")) as t:
        time.sleep(0.05)
    assert 40 < t.elapsed_ms < 200


def test_timer_returns_self():
    ctx = TimerContext(device=torch.device("cpu"))
    with ctx as t:
        pass
    assert t is ctx


def test_timer_initial_elapsed_zero():
    t = TimerContext(device=torch.device("cpu"))
    assert t.elapsed_ms == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_timer_cuda_elapsed_positive():
    with TimerContext(device=torch.device("cuda")) as t:
        torch.randn(1000, device="cuda")
    assert t.elapsed_ms > 0
