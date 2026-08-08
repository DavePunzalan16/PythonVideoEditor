"""
Frame-level video restoration processing.

Applies a sequence of corrections to a single BGR frame to reverse
common "beauty filter" / posterization effects:
  1. Optional horizontal un-mirror
  2. Light bilateral smoothing to reduce posterization blockiness
  3. LAB-space de-tinting (removes green/yellow color cast)
  4. CLAHE adaptive contrast on the L channel
  5. HSV saturation boost
  6. Unsharp-mask sharpening

IMPORTANT: The bilateral filter is intentionally light — too much smoothing
destroys facial features and makes the output look worse than the input.
The goal is subtle de-blocking, NOT aggressive smoothing.

All numeric parameters are exposed as module-level constants for easy tuning.
"""

import numpy as np
import cv2

# ─── Tunable Parameters ──────────────────────────────────────────────────────

# Bilateral filter: LIGHT pass to reduce posterization edges.
# Only applied ONCE with conservative settings to preserve face detail.
BILATERAL_D = 7                 # Pixel neighborhood diameter (smaller = faster + preserves detail)
BILATERAL_SIGMA_COLOR = 50      # Filter sigma in color space (lower = preserves edges better)
BILATERAL_SIGMA_SPACE = 50      # Filter sigma in coordinate space

# De-tint: shifts LAB A/B channels toward neutral (128).
# 1.0 = full neutralization, 0.0 = no change.
DETINT_STRENGTH = 0.9

# CLAHE: adaptive histogram equalization on L channel.
CLAHE_CLIP_LIMIT = 2.0         # Contrast limiting threshold
CLAHE_GRID_SIZE = (8, 8)       # Tile grid size for local adaptation

# Saturation boost: multiplier on HSV S channel.
SATURATION_MULTIPLIER = 1.35

# Unsharp mask: sharpening via Gaussian blur difference.
SHARPEN_KERNEL_SIZE = (3, 3)   # Gaussian blur kernel for the mask (smaller = finer detail)
SHARPEN_AMOUNT = 0.8           # Strength of sharpening (reduced to avoid artifacts)


# ─── Processing Functions ────────────────────────────────────────────────────

def process_frame(
    frame: np.ndarray,
    unmirror: bool = False,
    detint_strength: float = DETINT_STRENGTH,
    clahe_clip_limit: float = CLAHE_CLIP_LIMIT,
    saturation_multiplier: float = SATURATION_MULTIPLIER,
    sharpen_amount: float = SHARPEN_AMOUNT,
) -> np.ndarray:
    """
    Apply the full restoration pipeline to a single BGR frame.

    Parameters
    ----------
    frame : np.ndarray
        Input image in BGR color space (as returned by cv2.imread).
    unmirror : bool
        If True, flip the frame horizontally before processing.
    detint_strength : float
        How aggressively to neutralize color cast (0.0–1.0).
    clahe_clip_limit : float
        CLAHE contrast clip limit.
    saturation_multiplier : float
        Factor to multiply saturation channel by.
    sharpen_amount : float
        Unsharp mask strength.

    Returns
    -------
    np.ndarray
        Processed BGR frame with same dimensions as input.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Empty or None frame passed to process_frame")

    result = frame.copy()

    # Step 1: Optional un-mirror
    if unmirror:
        result = cv2.flip(result, 1)  # 1 = horizontal flip

    # Step 2: LIGHT bilateral smoothing (ONE pass only — preserves facial detail)
    result = cv2.bilateralFilter(
        result, BILATERAL_D, BILATERAL_SIGMA_COLOR, BILATERAL_SIGMA_SPACE
    )

    # Step 3: LAB de-tinting
    result = _detint_lab(result, detint_strength)

    # Step 4: CLAHE contrast on L channel
    result = _apply_clahe(result, clahe_clip_limit)

    # Step 5: Saturation boost
    result = _boost_saturation(result, saturation_multiplier)

    # Step 6: Unsharp mask sharpening
    result = _unsharp_mask(result, sharpen_amount)

    return result


def _detint_lab(frame: np.ndarray, strength: float) -> np.ndarray:
    """
    Remove color cast by shifting LAB A/B channels toward neutral.

    The average of A and B channels represents the overall tint.
    We shift each pixel's A and B values toward 128 (neutral gray)
    by the given strength factor.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute channel means
    a_mean = lab[:, :, 1].mean()
    b_mean = lab[:, :, 2].mean()

    # Shift toward neutral (128)
    lab[:, :, 1] -= (a_mean - 128) * strength
    lab[:, :, 2] -= (b_mean - 128) * strength

    # Clip to valid range and convert back
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _apply_clahe(frame: np.ndarray, clip_limit: float) -> np.ndarray:
    """
    Apply CLAHE to the L channel in LAB space to restore contrast.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit, tileGridSize=CLAHE_GRID_SIZE
    )
    lab[:, :, 0] = clahe.apply(l_channel)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _boost_saturation(frame: np.ndarray, multiplier: float) -> np.ndarray:
    """
    Boost saturation by multiplying the S channel in HSV space.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= multiplier
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _unsharp_mask(frame: np.ndarray, amount: float) -> np.ndarray:
    """
    Sharpen using unsharp masking: frame + amount * (frame - blurred).
    """
    if amount <= 0:
        return frame

    blurred = cv2.GaussianBlur(frame, SHARPEN_KERNEL_SIZE, 0)
    # Compute in float to avoid overflow
    sharpened = frame.astype(np.float32) + amount * (
        frame.astype(np.float32) - blurred.astype(np.float32)
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)
