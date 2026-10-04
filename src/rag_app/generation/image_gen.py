"""Optional 'artistic render' of an architecture using a HuggingFace diffusion model.

This is a flavor feature — the structured Mermaid diagram is the accurate output.
The model (SD-Turbo) is loaded lazily on first use and cached, so nothing is
downloaded until the user actually clicks 'Generate AI art view'.

Needs: pip install '.[image]'  (diffusers, accelerate)
"""
import base64
import io

from rag_app.compute import resolve_device
from rag_app.config import get_settings
from rag_app.logging_utils import get_logger

log = get_logger(__name__)

_PIPE = None
_MODEL = "stabilityai/sd-turbo"  # fast: good quality in ~1-2 steps


def _get_pipe():
    global _PIPE
    if _PIPE is None:
        from diffusers import AutoPipelineForText2Image

        s = get_settings()
        device = resolve_device(s)
        log.info("Loading diffusion model %s on %s (first time downloads ~2.5GB)", _MODEL, device)
        import torch

        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = AutoPipelineForText2Image.from_pretrained(_MODEL, torch_dtype=dtype, token=s.hf_token)
        _PIPE = pipe.to(device)
    return _PIPE


def _prompt_for(question: str, services: list[str]) -> str:
    svc = ", ".join(services[:6]) if services else "cloud services"
    return (
        f"A clean isometric cloud architecture diagram illustrating: {question}. "
        f"Featuring {svc}. Flat vector style, labeled boxes and arrows, soft colors, "
        f"professional infographic, high detail."
    )


def generate_architecture_image(question: str, services: list[str], steps: int = 2) -> str:
    """Return a base64 PNG data URI for an AI-rendered architecture illustration."""
    pipe = _get_pipe()
    prompt = _prompt_for(question, services)
    image = pipe(prompt=prompt, num_inference_steps=max(1, steps), guidance_scale=0.0).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
