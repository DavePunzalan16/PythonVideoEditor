"""
AI Face Restoration module using GFPGAN.

This generates a plausible, natural-looking human face from heavily
degraded/posterized input. It is a RECONSTRUCTION — not a guaranteed
match to the real person's exact features, since that data no longer
exists in the filtered file.

HARD BOUNDARY: This module is generic/unconditioned only. It does NOT
accept reference photos of specific people. That would be identity-swap
functionality (deepfake territory) and is explicitly out of scope.

Requirements:
    pip install gfpgan torch torchvision facexlib basicsr

GPU recommended for practical speed. CPU works but is very slow.
"""

import logging
import numpy as np
import cv2
from typing import Optional

logger = logging.getLogger(__name__)

# Track availability
_GFPGAN_AVAILABLE = False
_restorer = None

try:
    from gfpgan import GFPGANer
    _GFPGAN_AVAILABLE = True
except ImportError:
    # Try alternative: gfpgan may work without explicit basicsr import
    # if torch and facexlib are present
    try:
        import torch
        import facexlib
        from gfpgan import GFPGANer
        _GFPGAN_AVAILABLE = True
    except ImportError:
        logger.info(
            "GFPGAN not installed or dependencies missing. AI face restoration unavailable. "
            "Install: pip install gfpgan torch torchvision facexlib"
        )


# ─── Skin Tone Constants ─────────────────────────────────────────────────────

# Realistic skin tone ranges in HSV (OpenCV scale: H=0-180, S=0-255, V=0-255)
SKIN_HUE_MIN = 0
SKIN_HUE_MAX = 50       # Skin typically falls in 0-50 hue range
SKIN_SAT_MIN = 20
SKIN_SAT_MAX = 180      # Not too desaturated, not neon
SKIN_VAL_MIN = 50
SKIN_VAL_MAX = 245


def is_face_restore_available() -> bool:
    """Check if GFPGAN dependencies are installed."""
    return _GFPGAN_AVAILABLE


def _get_restorer(upscale: int = 1):
    """Initialize and cache the GFPGAN restorer."""
    global _restorer

    if not _GFPGAN_AVAILABLE:
        raise RuntimeError(
            "GFPGAN is not installed. "
            "Run: pip install gfpgan torch torchvision facexlib basicsr"
        )

    if _restorer is not None:
        return _restorer

    import torch

    # GFPGAN v1.4 model — best quality for face restoration
    _restorer = GFPGANer(
        model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        upscale=upscale,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,  # No background upsampling (keeps original bg)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"GFPGAN initialized on {device}")
    return _restorer


def restore_face(
    frame: np.ndarray,
    blend_weight: float = 0.7,
    upscale: int = 1,
) -> np.ndarray:
    """
    Run GFPGAN face restoration on a single frame.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame.
    blend_weight : float
        Blend between restored face and original.
        0.0 = original only, 1.0 = full AI restoration.
        Default 0.7 for natural look.
    upscale : int
        Upscale factor (1 = same resolution, 2 = 2x).

    Returns
    -------
    np.ndarray
        Frame with faces restored (same dimensions as input if upscale=1).
    """
    if not _GFPGAN_AVAILABLE:
        logger.warning("GFPGAN not available, returning original frame.")
        return frame

    restorer = _get_restorer(upscale=upscale)

    try:
        # GFPGAN returns: cropped_faces, restored_faces, restored_img
        # restored_img is the full frame with faces replaced
        _, _, restored_img = restorer.enhance(
            frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True,
            weight=blend_weight,
        )

        if restored_img is None:
            # No face detected or restoration failed
            logger.debug("No face detected in frame, returning original.")
            return frame

        # Ensure output matches input dimensions
        h, w = frame.shape[:2]
        rh, rw = restored_img.shape[:2]
        if (rh, rw) != (h, w):
            restored_img = cv2.resize(restored_img, (w, h), interpolation=cv2.INTER_LANCZOS4)

        return restored_img

    except Exception as e:
        logger.warning(f"Face restoration failed: {e}")
        return frame


def normalize_skin_tone(frame: np.ndarray, face_region: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize skin tones to fall within realistic human range.

    After face restoration + saturation boost, skin can look unnatural
    (too green, too yellow, too saturated). This nudges skin-colored
    pixels back into a natural range.

    Parameters
    ----------
    frame : np.ndarray
        BGR frame (post face-restoration).
    face_region : np.ndarray, optional
        Mask of face region. If None, detects skin by color range.

    Returns
    -------
    np.ndarray
        Frame with normalized skin tones.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Create a skin mask based on HSV ranges
    # This catches pixels that SHOULD be skin but have wrong color
    # We use a broader range to catch green/yellow shifted skin
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Detect pixels that are likely skin by value/saturation but wrong hue
    # (e.g., green-shifted skin has hue 50-90 instead of 0-50)
    likely_skin = (
        (v_channel > 60) & (v_channel < 240) &
        (s_channel > 15) & (s_channel < 200)
    )

    # Pixels with hue in green/yellow zone that should be skin
    green_shifted = likely_skin & (h_channel > 35) & (h_channel < 95)

    # Shift green-shifted skin back toward natural skin hue range
    # Natural skin hue center is around 15-20 (orange/peach)
    if np.any(green_shifted):
        # Gradually pull hue toward skin range
        correction = (h_channel[green_shifted] - 20) * 0.6
        hsv[:, :, 0][green_shifted] -= correction
        hsv[:, :, 0] = np.clip(hsv[:, :, 0], 0, 179)

        # Also slightly reduce saturation on corrected areas to avoid neon look
        hsv[:, :, 1][green_shifted] *= 0.85
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    result = np.clip(hsv, 0, [179, 255, 255]).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)


def restore_and_normalize(
    frame: np.ndarray,
    blend_weight: float = 0.7,
    normalize_skin: bool = True,
) -> np.ndarray:
    """
    Full face restoration pipeline: GFPGAN + skin tone normalization.

    Parameters
    ----------
    frame : np.ndarray
        Input BGR frame (should already be color-corrected).
    blend_weight : float
        GFPGAN blend weight (0.0-1.0).
    normalize_skin : bool
        Whether to apply skin tone normalization after restoration.

    Returns
    -------
    np.ndarray
        Frame with restored face and natural skin tones.
    """
    # Step 1: AI face restoration
    result = restore_face(frame, blend_weight=blend_weight)

    # Step 2: Skin tone normalization (fix green/yellow skin)
    if normalize_skin:
        result = normalize_skin_tone(result)

    return result
