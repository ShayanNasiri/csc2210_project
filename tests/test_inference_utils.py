"""Tests for src/inference_utils.py — load_tokenized_data, get_batch_slice, BatchRunner."""

import pytest
import torch

from src.inference_utils import BatchRunner, get_batch_slice, load_tokenized_data


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_data():
    """Synthetic tokenized data dict matching the real schema."""
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(0, 1000, (20, 32)),
        "attention_mask": torch.ones(20, 32, dtype=torch.long),
        "token_type_ids": torch.zeros(20, 32, dtype=torch.long),
        "qids": [f"q{i}" for i in range(20)],
        "labels": [0] * 10 + [1] * 10,
    }


@pytest.fixture
def fake_data_no_tids():
    """Synthetic data without token_type_ids."""
    torch.manual_seed(0)
    return {
        "input_ids": torch.randint(0, 1000, (20, 32)),
        "attention_mask": torch.ones(20, 32, dtype=torch.long),
        "qids": [f"q{i}" for i in range(20)],
        "labels": [0] * 10 + [1] * 10,
    }


# ---------------------------------------------------------------------------
# load_tokenized_data tests
# ---------------------------------------------------------------------------

class TestLoadTokenizedData:

    def test_roundtrip(self, tmp_path, fake_data):
        path = str(tmp_path / "test_data.pt")
        torch.save(fake_data, path)
        loaded = load_tokenized_data(path)

        assert set(loaded.keys()) == {"input_ids", "attention_mask", "token_type_ids", "qids", "labels"}
        assert torch.equal(loaded["input_ids"], fake_data["input_ids"])
        assert torch.equal(loaded["attention_mask"], fake_data["attention_mask"])
        assert torch.equal(loaded["token_type_ids"], fake_data["token_type_ids"])
        assert loaded["qids"] == fake_data["qids"]
        assert loaded["labels"] == fake_data["labels"]

    def test_missing_file_raises(self):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_tokenized_data("/nonexistent/path/data.pt")

    def test_missing_token_type_ids_returns_none(self, tmp_path, fake_data_no_tids):
        path = str(tmp_path / "no_tids.pt")
        torch.save(fake_data_no_tids, path)
        loaded = load_tokenized_data(path)

        assert loaded["token_type_ids"] is None
        assert torch.equal(loaded["input_ids"], fake_data_no_tids["input_ids"])


# ---------------------------------------------------------------------------
# get_batch_slice tests
# ---------------------------------------------------------------------------

class TestGetBatchSlice:

    def test_shape(self, fake_data):
        ids, mask, tids = get_batch_slice(fake_data, 0, 4, torch.device("cpu"))
        assert ids.shape == (4, 32)
        assert mask.shape == (4, 32)
        assert tids.shape == (4, 32)

    def test_last_batch(self, fake_data):
        ids, mask, tids = get_batch_slice(fake_data, 16, 20, torch.device("cpu"))
        assert ids.shape == (4, 32)
        assert torch.equal(ids, fake_data["input_ids"][16:20])

    def test_device(self, fake_data):
        device = torch.device("cpu")
        ids, mask, tids = get_batch_slice(fake_data, 0, 4, device)
        assert ids.device == device
        assert mask.device == device
        assert tids.device == device

    def test_no_token_type_ids(self, fake_data):
        data_no_tids = dict(fake_data)
        data_no_tids["token_type_ids"] = None
        ids, mask, tids = get_batch_slice(data_no_tids, 0, 4, torch.device("cpu"))
        assert tids is None

    def test_values_match_source(self, fake_data):
        ids, mask, tids = get_batch_slice(fake_data, 2, 5, torch.device("cpu"))
        assert torch.equal(ids, fake_data["input_ids"][2:5])
        assert torch.equal(mask, fake_data["attention_mask"][2:5])
        assert torch.equal(tids, fake_data["token_type_ids"][2:5])


# ---------------------------------------------------------------------------
# BatchRunner tests
# ---------------------------------------------------------------------------

class TestBatchRunner:

    def test_num_batches_ceil(self):
        runner = BatchRunner(num_samples=20, batch_size=8)
        assert runner.num_batches == 3  # ceil(20/8)

    def test_num_batches_exact(self):
        runner = BatchRunner(num_samples=16, batch_size=8)
        assert runner.num_batches == 2

    def test_warmup_clamped(self):
        runner = BatchRunner(num_samples=20, batch_size=8, warmup_batches=100)
        assert runner.warmup_batches == 3  # clamped to num_batches

    def test_timed_clamped(self):
        runner = BatchRunner(num_samples=20, batch_size=8, timed_batch_limit=50)
        assert runner.timed_batches == 3  # clamped to num_batches

    def test_warmup_calls_forward(self, fake_data):
        call_count = [0]

        def counting_fn(ids, mask, tids):
            call_count[0] += 1

        runner = BatchRunner(num_samples=20, batch_size=8, warmup_batches=2)
        runner.warmup(fake_data, torch.device("cpu"), counting_fn)
        assert call_count[0] == 2

    def test_run_processes_all_batches(self, fake_data):
        def identity_fn(ids, mask, tids):
            return ids.shape[0]

        runner = BatchRunner(num_samples=20, batch_size=8, warmup_batches=0, timed_batch_limit=10)
        outputs, latencies = runner.run_with_timing(fake_data, torch.device("cpu"), identity_fn)
        assert len(outputs) == 3  # ceil(20/8) = 3 batches

    def test_run_returns_correct_latency_count(self, fake_data):
        def identity_fn(ids, mask, tids):
            return ids.shape[0]

        runner = BatchRunner(num_samples=20, batch_size=8, warmup_batches=0, timed_batch_limit=2)
        outputs, latencies = runner.run_with_timing(fake_data, torch.device("cpu"), identity_fn)
        assert len(latencies) == 2  # timed_batch_limit=2, num_batches=3

    def test_latencies_positive(self, fake_data):
        def identity_fn(ids, mask, tids):
            return ids.shape[0]

        runner = BatchRunner(num_samples=20, batch_size=8, warmup_batches=0, timed_batch_limit=10)
        _, latencies = runner.run_with_timing(fake_data, torch.device("cpu"), identity_fn)
        assert all(lat > 0 for lat in latencies)

    def test_batch_sizes_correct(self, fake_data):
        """Verify batch sizes: first two are 8, last is 4 (20 samples total)."""
        batch_sizes = []

        def record_fn(ids, mask, tids):
            batch_sizes.append(ids.shape[0])
            return ids.shape[0]

        runner = BatchRunner(num_samples=20, batch_size=8, warmup_batches=0)
        runner.run_with_timing(fake_data, torch.device("cpu"), record_fn)
        assert batch_sizes == [8, 8, 4]
