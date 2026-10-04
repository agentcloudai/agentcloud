"""Shared compute helpers (device selection for GPU/CPU)."""
from rag_app.config import Settings
from rag_app.logging_utils import get_logger

log = get_logger(__name__)


def resolve_device(s: Settings) -> str:
    """Return the torch device to use: honour settings.device, "auto" picks CUDA if present."""
    want = (s.device or "auto").lower()
    if want == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001 - torch should be present, but never hard-fail here
            device = "cpu"
    else:
        device = want
    log.info("Using compute device: %s", device)
    return device
