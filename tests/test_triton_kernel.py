"""Tests for Phase 5: Triton batch-compaction kernel (src/triton_compact.py).

All tests run on CPU (PyTorch fallback path) by default.
Float16 precision test requires CUDA and is skipped otherwise.
"""

import pytest
import torch

from src.triton_compact import (
    HAS_TRITON,
    compact_batch,
    compute_compaction_indices,
    scatter_scores_back,
)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

# Canonical test mask: selects rows 0, 2, 3, 6 from a batch of 8
CANONICAL_MASK = torch.tensor([True, False, True, True, False, False, True, False])
CANONICAL_ACTIVE_ROWS = [0, 2, 3, 6]
CANONICAL_B = 8
CANONICAL_NEW_B = 4


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Test 1: compact_batch correctness on hidden states
# ---------------------------------------------------------------------------

class TestCompactCorrectness:
    """Verify that compact_batch selects the correct rows from hidden_states."""

    S = 16
    H = 384

    def test_compact_selects_active_rows(self):
        device = _device()
        B = CANONICAL_B
        # Fill row i with float(i) so we can verify which rows survived.
        hidden = torch.zeros(B, self.S, self.H, device=device)
        for i in range(B):
            hidden[i] = float(i)
        attn = torch.ones(B, self.S, device=device)
        mask = CANONICAL_MASK.to(device)

        compacted_h, compacted_a, indices = compact_batch(hidden, attn, mask)

        assert compacted_h.shape == (CANONICAL_NEW_B, self.S, self.H), (
            f"Expected shape ({CANONICAL_NEW_B}, {self.S}, {self.H}), "
            f"got {compacted_h.shape}"
        )
        for out_row, src_row in enumerate(CANONICAL_ACTIVE_ROWS):
            expected_val = float(src_row)
            actual_val = compacted_h[out_row, 0, 0].item()
            assert actual_val == pytest.approx(expected_val), (
                f"Row {out_row} should contain src row {src_row} "
                f"(value {expected_val}), got {actual_val}"
            )


# ---------------------------------------------------------------------------
# Test 2: compact_batch correctness on attention mask
# ---------------------------------------------------------------------------

class TestCompactAttentionMask:
    """Verify that compact_batch selects the correct rows from attention_mask."""

    S = 16

    def test_attention_mask_rows(self):
        device = _device()
        B = CANONICAL_B
        # Row i has all values set to float(i).
        attn = torch.zeros(B, self.S, device=device)
        for i in range(B):
            attn[i] = float(i)
        hidden = torch.zeros(B, self.S, 384, device=device)
        mask = CANONICAL_MASK.to(device)

        _, compacted_a, _ = compact_batch(hidden, attn, mask)

        assert compacted_a.shape == (CANONICAL_NEW_B, self.S), (
            f"Expected shape ({CANONICAL_NEW_B}, {self.S}), got {compacted_a.shape}"
        )
        for out_row, src_row in enumerate(CANONICAL_ACTIVE_ROWS):
            expected_val = float(src_row)
            actual_val = compacted_a[out_row, 0].item()
            assert actual_val == pytest.approx(expected_val), (
                f"Attn row {out_row} should contain src row {src_row} "
                f"(value {expected_val}), got {actual_val}"
            )


# ---------------------------------------------------------------------------
# Test 3: scatter_scores_back
# ---------------------------------------------------------------------------

class TestScatterBack:
    """Verify scatter_scores_back places compacted scores at active positions."""

    def test_scatter_fills_active_zeros_inactive(self):
        device = _device()
        mask = CANONICAL_MASK.to(device)
        indices, new_b = compute_compaction_indices(mask)

        compacted_scores = torch.tensor([0.9, 0.8, 0.7, 0.6], device=device)
        full = scatter_scores_back(compacted_scores, indices, CANONICAL_B, mask)

        assert full.shape == (CANONICAL_B,), f"Expected shape ({CANONICAL_B},), got {full.shape}"

        # Active positions should have the assigned scores.
        expected_active = {0: 0.9, 2: 0.8, 3: 0.7, 6: 0.6}
        for pos, score in expected_active.items():
            assert full[pos].item() == pytest.approx(score, abs=1e-6), (
                f"Position {pos} should be {score}, got {full[pos].item()}"
            )

        # Inactive positions should be zero.
        inactive = [1, 4, 5, 7]
        for pos in inactive:
            assert full[pos].item() == pytest.approx(0.0), (
                f"Inactive position {pos} should be 0.0, got {full[pos].item()}"
            )


# ---------------------------------------------------------------------------
# Test 4: numerical precision (float32 + float16)
# ---------------------------------------------------------------------------

class TestNumericalPrecision:
    """Verify compact_batch matches a PyTorch reference gather."""

    B, S, H = 8, 16, 384

    def _reference_compact(self, hidden, attn, mask):
        """Pure PyTorch reference: boolean indexing."""
        return hidden[mask], attn[mask]

    def test_float32_precision(self):
        device = _device()
        torch.manual_seed(123)
        hidden = torch.randn(self.B, self.S, self.H, device=device)
        attn = torch.randn(self.B, self.S, device=device)
        mask = CANONICAL_MASK.to(device)

        compacted_h, compacted_a, _ = compact_batch(hidden, attn, mask)
        ref_h, ref_a = self._reference_compact(hidden, attn, mask)

        assert torch.allclose(compacted_h, ref_h, atol=1e-6), (
            f"Float32 hidden mismatch, max diff: "
            f"{(compacted_h - ref_h).abs().max().item():.2e}"
        )
        assert torch.allclose(compacted_a, ref_a, atol=1e-6), (
            f"Float32 attn mismatch, max diff: "
            f"{(compacted_a - ref_a).abs().max().item():.2e}"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for float16")
    def test_float16_bit_exact(self):
        device = torch.device("cuda")
        torch.manual_seed(123)
        hidden = torch.randn(self.B, self.S, self.H, device=device, dtype=torch.float16)
        attn = torch.randn(self.B, self.S, device=device, dtype=torch.float16)
        mask = CANONICAL_MASK.to(device)

        compacted_h, compacted_a, _ = compact_batch(hidden, attn, mask)
        ref_h, ref_a = self._reference_compact(hidden, attn, mask)

        assert torch.equal(compacted_h, ref_h), (
            f"Float16 hidden not bit-exact, max diff: "
            f"{(compacted_h - ref_h).abs().max().item():.2e}"
        )
        assert torch.equal(compacted_a, ref_a), (
            f"Float16 attn not bit-exact, max diff: "
            f"{(compacted_a - ref_a).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Test 5: all-active mask (identity compaction)
# ---------------------------------------------------------------------------

class TestAllActive:
    """When all docs are active, output must equal input."""

    B, S, H = 8, 16, 384

    def test_all_active_identity(self):
        device = _device()
        torch.manual_seed(99)
        hidden = torch.randn(self.B, self.S, self.H, device=device)
        attn = torch.randn(self.B, self.S, device=device)
        mask = torch.ones(self.B, dtype=torch.bool, device=device)

        compacted_h, compacted_a, indices = compact_batch(hidden, attn, mask)

        assert compacted_h.shape == hidden.shape, (
            f"All-active: shape should be unchanged, got {compacted_h.shape}"
        )
        assert torch.equal(compacted_h, hidden), "All-active: hidden should be identical"
        assert torch.equal(compacted_a, attn), "All-active: attn should be identical"


# ---------------------------------------------------------------------------
# Test 6: all-exited mask (empty output)
# ---------------------------------------------------------------------------

class TestAllExited:
    """When no docs are active, output batch_size must be 0."""

    B, S, H = 8, 16, 384

    def test_all_exited_empty(self):
        device = _device()
        hidden = torch.randn(self.B, self.S, self.H, device=device)
        attn = torch.randn(self.B, self.S, device=device)
        mask = torch.zeros(self.B, dtype=torch.bool, device=device)

        compacted_h, compacted_a, indices = compact_batch(hidden, attn, mask)

        assert compacted_h.shape[0] == 0, (
            f"All-exited: batch dim should be 0, got {compacted_h.shape[0]}"
        )
        assert compacted_a.shape[0] == 0, (
            f"All-exited: attn batch dim should be 0, got {compacted_a.shape[0]}"
        )
        assert compacted_h.shape[1:] == (self.S, self.H), (
            f"All-exited: hidden trailing dims should be ({self.S}, {self.H})"
        )
        assert compacted_a.shape[1:] == (self.S,), (
            f"All-exited: attn trailing dims should be ({self.S},)"
        )


# ---------------------------------------------------------------------------
# Test 7: large batch stress test
# ---------------------------------------------------------------------------

class TestLargeBatch:
    """B=1024, S=128, H=384, ~70% active — verify shape and correctness."""

    B, S, H = 1024, 128, 384

    def test_large_batch_shape_and_values(self):
        device = _device()
        torch.manual_seed(42)
        mask = torch.rand(self.B, device=device) > 0.3  # ~70% active
        expected_new_b = mask.sum().item()

        hidden = torch.randn(self.B, self.S, self.H, device=device)
        attn = torch.randn(self.B, self.S, device=device)

        compacted_h, compacted_a, indices = compact_batch(hidden, attn, mask)

        assert compacted_h.shape == (expected_new_b, self.S, self.H), (
            f"Large batch: expected ({expected_new_b}, {self.S}, {self.H}), "
            f"got {compacted_h.shape}"
        )
        assert compacted_a.shape == (expected_new_b, self.S), (
            f"Large batch: expected ({expected_new_b}, {self.S}), "
            f"got {compacted_a.shape}"
        )

        # Verify values match PyTorch boolean indexing reference.
        ref_h = hidden[mask]
        ref_a = attn[mask]
        assert torch.allclose(compacted_h, ref_h, atol=1e-6), (
            f"Large batch: hidden mismatch, max diff: "
            f"{(compacted_h - ref_h).abs().max().item():.2e}"
        )
        assert torch.allclose(compacted_a, ref_a, atol=1e-6), (
            f"Large batch: attn mismatch, max diff: "
            f"{(compacted_a - ref_a).abs().max().item():.2e}"
        )


# ---------------------------------------------------------------------------
# Supplementary: compute_compaction_indices unit test
# ---------------------------------------------------------------------------

class TestComputeCompactionIndices:
    """Verify compute_compaction_indices returns correct scatter indices."""

    def test_canonical_indices(self):
        device = _device()
        mask = CANONICAL_MASK.to(device)
        indices, new_b = compute_compaction_indices(mask)

        assert new_b == CANONICAL_NEW_B
        # Expected: cumsum([1,0,1,1,0,0,1,0]) - 1 = [0,-1,1,2,-1,-1,3,-1]
        # (inactive positions get cumsum value of previous active, minus 1 is
        #  meaningless for them but deterministic)
        expected = torch.cumsum(mask.int(), 0) - 1
        assert torch.equal(indices.cpu(), expected.cpu()), (
            f"Indices mismatch: expected {expected.tolist()}, got {indices.cpu().tolist()}"
        )
