"""Phase 5: Triton batch-compaction kernel with PyTorch fallback.

Provides functions to physically compact a batch by removing exited documents,
and to scatter scores back to their original positions.
"""

import torch

# Triton is Linux-only; gracefully degrade on other platforms.
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _compact_kernel(
        src_ptr,
        dst_ptr,
        src_indices_ptr,
        n_elements_per_row: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Copy one block of elements from a source row to a destination row.

        Grid: (new_batch_size, ceil(n_elements_per_row / BLOCK_SIZE))
        """
        dst_row = tl.program_id(0)
        block_idx = tl.program_id(1)

        # Which source row to read from
        src_row = tl.load(src_indices_ptr + dst_row)

        # Offsets within the row
        offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements_per_row

        # Compute flat addresses
        src_addr = src_row * n_elements_per_row + offsets
        dst_addr = dst_row * n_elements_per_row + offsets

        # Load and store
        vals = tl.load(src_ptr + src_addr, mask=mask)
        tl.store(dst_ptr + dst_addr, vals, mask=mask)


# ---------------------------------------------------------------------------
# Pure PyTorch helpers
# ---------------------------------------------------------------------------


def compute_compaction_indices(active_mask: torch.Tensor):
    """Compute scatter indices and new batch size from an active mask.

    Args:
        active_mask: (batch,) bool tensor

    Returns:
        scatter_indices: (batch,) int tensor — cumsum-1 of active_mask
        new_batch_size: int
    """
    scatter_indices = torch.cumsum(active_mask.int(), dim=0) - 1
    new_batch_size = int(active_mask.sum().item())
    return scatter_indices, new_batch_size


def scatter_scores_back(
    compacted_scores: torch.Tensor,
    scatter_indices: torch.Tensor,
    original_batch_size: int,
    active_mask: torch.Tensor,
):
    """Place compacted scores back into a full-size tensor.

    Args:
        compacted_scores: (new_batch,) scores for active documents
        scatter_indices: (original_batch,) from compute_compaction_indices (unused)
        original_batch_size: original batch dimension
        active_mask: (original_batch,) bool tensor

    Returns:
        full_scores: (original_batch,) with zeros at inactive positions
    """
    full_scores = compacted_scores.new_zeros(original_batch_size)
    full_scores[active_mask] = compacted_scores
    return full_scores


# ---------------------------------------------------------------------------
# compact_batch: Triton on CUDA, PyTorch fallback on CPU / no-Triton
# ---------------------------------------------------------------------------

BLOCK_SIZE = 1024


def compact_batch(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    active_mask: torch.Tensor,
):
    """Compact a batch by removing inactive (exited) rows.

    Args:
        hidden_states: (B, S, H)
        attention_mask: (B, S)
        active_mask: (B,) bool tensor

    Returns:
        compacted_hidden: (new_B, S, H)
        compacted_attn: (new_B, S)
        scatter_indices: (B,) int tensor from compute_compaction_indices
    """
    scatter_indices, new_batch_size = compute_compaction_indices(active_mask)

    # Edge case: nothing active
    if new_batch_size == 0:
        S = hidden_states.shape[1]
        H = hidden_states.shape[2]
        empty_h = hidden_states.new_empty(0, S, H)
        empty_a = attention_mask.new_empty(0, S)
        return empty_h, empty_a, scatter_indices

    # Build src_indices: shape (new_batch_size,), maps dst_row -> src_row
    src_indices = torch.where(active_mask)[0].to(dtype=torch.int64)

    use_triton = HAS_TRITON and hidden_states.is_cuda

    if use_triton:
        try:
            compacted_h = _triton_compact_tensor(
                hidden_states, src_indices, new_batch_size
            )
            compacted_a = _triton_compact_tensor(
                attention_mask, src_indices, new_batch_size
            )
            return compacted_h, compacted_a, scatter_indices
        except Exception:
            # Fall through to PyTorch fallback
            pass

    # PyTorch fallback
    compacted_h = hidden_states[src_indices].contiguous()
    compacted_a = attention_mask[src_indices].contiguous()
    return compacted_h, compacted_a, scatter_indices


def _triton_compact_tensor(tensor, src_indices, new_batch_size):
    """Launch the Triton compaction kernel for a single tensor.

    Args:
        tensor: (B, ...) — 2D or 3D
        src_indices: (new_batch_size,) int64
        new_batch_size: int

    Returns:
        compacted: (new_batch_size, ...) contiguous
    """
    original_shape = tensor.shape
    # Flatten to 2D: (B, n_elements_per_row)
    flat = tensor.reshape(original_shape[0], -1).contiguous()
    n_elements_per_row = flat.shape[1]

    dst = torch.empty(
        (new_batch_size, n_elements_per_row),
        dtype=flat.dtype,
        device=flat.device,
    )

    n_blocks = (n_elements_per_row + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (new_batch_size, n_blocks)

    _compact_kernel[grid](
        flat,
        dst,
        src_indices,
        n_elements_per_row,
        BLOCK_SIZE,
    )

    # Reshape back to original trailing dims
    new_shape = (new_batch_size,) + original_shape[1:]
    return dst.reshape(new_shape)
