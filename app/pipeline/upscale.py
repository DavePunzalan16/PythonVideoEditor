"""
Optional AI upscaling module using Real-ESRGAN.

This module wraps Real-ESRGAN inference per frame. It requires:
  - torch
  - realesrgan (pip install realesrgan)
  - A GPU for practical speed (CPU is extremely slow)

If dependencies are not available, upscaling is gracefully disabled.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Track whether Real-ESRGAN is available
_ESRGAN_AVAILABLE = False
_upsampler = None

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import torch
    _ESRGAN_AVAILABLE = True
except ImportError:
    logger.info(
        "Real-ESRGAN dependencies not installed. "
        "AI upscaling will be unavailable. "
        "To enable: pip install realesrgan torch"
    )


def is_upscale_available() -> bool:
    """Check if AI upscaling dependencies are installed."""
    return _ESRGAN_AVAILABLE


def get_upsampler(scale: int = 2):
    """
    Initialize and cache the Real-ESRGAN upsampler.

    Parameters
    ----------
    scale : int
        Upscaling factor (2 or 4).
    """
    global _upsampler

    if not _ESRGAN_AVAILABLE:
        raise RuntimeError(
            "Real-ESRGAN is not installed. "
            "Run: pip install realesrgan torch"
        )

    if _upsampler is not None:
        return _upsampler

    # Use RealESRGAN_x2plus for 2x, RealESRGAN_x4plus for 4x
    if scale == 2:
        model_name = "RealESRGAN_x2plus"
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=2,
        )
    else:
        model_name = "RealESRGAN_x4plus"
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64,
            num_block=23, num_grow_ch=32, scale=4,
        )

    # Model weights are auto-downloaded by realesrgan
    _upsampler = RealESRGANer(
        scale=scale,
        model_path=None,  # Auto-download
        model=model,
        tile=400,         # Tile size for memory efficiency
        tile_pad=10,
        pre_pad=0,
        half=torch.cuda.is_available(),  # FP16 only on GPU
    )

    logger.info(f"Initialized {model_name} upsampler (scale={scale}x)")
    return _upsampler


def upscale_frame(frame: np.ndarray, scale: int = 2) -> np.ndarray:
    """
    Upscale a single BGR frame using Real-ESRGAN.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    scale : int
        Upscaling factor.

    Returns
    -------
    np.ndarray
        Upscaled BGR frame.
    """
    if not _ESRGAN_AVAILABLE:
        logger.warning("Real-ESRGAN not available, returning original frame.")
        return frame

    upsampler = get_upsampler(scale)

    try:
        # Real-ESRGAN expects BGR input (same as OpenCV)
        output, _ = upsampler.enhance(frame, outscale=scale)
        return output
    except Exception as e:
        logger.error(f"Upscaling failed for frame: {e}")
        return frame  # Return original on failure
