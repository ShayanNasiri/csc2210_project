"""Smoke test: validates CUDA, Triton, model loading, and src.utils on the cluster."""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` is importable
# (needed when running directly via `python scripts/smoke_test.py`)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    # 1. CUDA check
    import torch

    assert torch.cuda.is_available(), "CUDA is not available"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
    print(f"VRAM: {vram / 1e9:.1f} GB")

    # 2. Triton check — compile a trivial vector-add kernel (Linux only)
    if sys.platform == "linux":
        import triton
        import triton.language as tl

        print(f"Triton version: {triton.__version__}")

        @triton.jit
        def _add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid * BLOCK + tl.arange(0, BLOCK)
            mask = offs < n
            x = tl.load(x_ptr + offs, mask=mask)
            y = tl.load(y_ptr + offs, mask=mask)
            tl.store(out_ptr + offs, x + y, mask=mask)

        n = 256
        x = torch.randn(n, device="cuda")
        y = torch.randn(n, device="cuda")
        out = torch.empty(n, device="cuda")
        _add_kernel[(1,)](x, y, out, n, BLOCK=256)
        assert torch.allclose(out, x + y), "Triton kernel output mismatch"
        print("Triton kernel compiled and verified.")
    else:
        print(f"Triton skipped (not available on {sys.platform})")

    # 3. Load tokenizer and model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval().to("cuda")
    print(f"Model loaded: {model_name}")

    # 4. Dummy forward pass
    pairs = [
        ("what is python", "Python is a programming language."),
        ("capital of france", "Paris is the capital of France."),
        ("how to cook rice", "Boil water and add rice."),
        ("GPU computing", "GPUs accelerate parallel workloads."),
    ]
    inputs = tokenizer(
        [q for q, _ in pairs],
        [d for _, d in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**inputs).logits

    assert logits.shape == (4, 1), f"Expected logit shape (4, 1), got {logits.shape}"
    print(f"Forward pass OK, logit shape: {logits.shape}")

    # 5. Verify src.utils
    from src.utils import TimerContext, get_device, set_seed

    set_seed()
    device = get_device()
    assert device.type == "cuda", f"Expected cuda, got {device.type}"

    with TimerContext(device) as t:
        torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
    assert t.elapsed_ms > 0, "TimerContext did not record time"
    print(f"src.utils OK (device={device}, timer={t.elapsed_ms:.2f}ms)")

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
