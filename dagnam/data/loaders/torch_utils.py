"""Small PyTorch loader utilities."""

from __future__ import annotations



def should_pin_memory() -> bool:
    """Return whether PyTorch pinned memory should be requested."""
    import torch
    accelerator = getattr(torch, "accelerator", None)
    is_accelerator_available = getattr(accelerator, "is_available", None)
    if callable(is_accelerator_available):
        try:
            return bool(is_accelerator_available())
        except Exception:
            return False

    cuda = getattr(torch, "cuda", None)
    is_cuda_available = getattr(cuda, "is_available", None)
    if callable(is_cuda_available) and is_cuda_available():
        return True

    xpu = getattr(torch, "xpu", None)
    is_xpu_available = getattr(xpu, "is_available", None)
    if callable(is_xpu_available) and is_xpu_available():
        return True

    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    is_mps_available = getattr(mps, "is_available", None)
    return bool(callable(is_mps_available) and is_mps_available())
