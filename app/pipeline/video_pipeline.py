"""
Video processing pipeline for filter restoration.

Orchestrates the full flow:
  1. Extract frames from input video (ffmpeg)
  2. Extract audio track (ffmpeg)
  3. Process each frame through the restoration pipeline
  4. Reassemble processed frames + audio into output video (ffmpeg)
  5. Clean up temporary files

All ffmpeg calls have error handling and produce clear messages on failure.
"""

import os
import sys
import glob
import shutil
import logging
import subprocess
from typing import Optional, Callable

import numpy as np
import cv2

# Add parent directories so we can import sibling modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.pipeline.frame_processor import process_frame

logger = logging.getLogger(__name__)

# Supported video formats
SUPPORTED_FORMATS = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system PATH."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_video_info(video_path: str) -> dict:
    """Extract frame rate, width, and height from a video file using ffprobe."""
    info = {"framerate": 30.0, "width": None, "height": None}
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,width,height",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(f"ffprobe failed, using defaults: {result.stderr}")
            return info

        # Output format: width,height,r_frame_rate (e.g. "1080,1920,30/1")
        parts = result.stdout.strip().split(",")
        if len(parts) >= 3:
            info["width"] = int(parts[0])
            info["height"] = int(parts[1])
            rate_str = parts[2]
            if "/" in rate_str:
                num, den = rate_str.split("/")
                info["framerate"] = float(num) / float(den)
            else:
                info["framerate"] = float(rate_str)
        elif len(parts) >= 1:
            rate_str = parts[-1]
            if "/" in rate_str:
                num, den = rate_str.split("/")
                info["framerate"] = float(num) / float(den)

    except Exception as e:
        logger.warning(f"Could not determine video info, using defaults: {e}")

    return info


def extract_frames(video_path: str, output_dir: str) -> dict:
    """
    Extract all frames from a video as PNG files.

    Returns dict with framerate, width, height.
    """
    os.makedirs(output_dir, exist_ok=True)

    video_info = get_video_info(video_path)
    logger.info(f"Detected: {video_info['framerate']:.2f} fps, {video_info['width']}x{video_info['height']}")

    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-fps_mode", "passthrough",
                os.path.join(output_dir, "frame_%06d.png"),
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg frame extraction failed:\n{result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Frame extraction timed out (>10 minutes).")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found on system. Please install ffmpeg. "
            "See SETUP.md for instructions."
        )

    frame_count = len(glob.glob(os.path.join(output_dir, "frame_*.png")))
    logger.info(f"Extracted {frame_count} frames to {output_dir}")
    return video_info


def extract_audio(video_path: str, output_path: str) -> Optional[str]:
    """
    Extract the audio track from a video file.

    Returns the output path if audio was extracted, None if video has no audio.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-i", video_path,
                "-vn",
                "-acodec", "copy",
                "-y",
                output_path,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            # Check if it's a "no audio" situation vs a real error
            if "does not contain any stream" in result.stderr or \
               "Output file #0 does not contain any stream" in result.stderr:
                logger.info("Video has no audio track, skipping audio extraction.")
                return None
            # Some videos legitimately have no audio
            logger.info(f"No audio extracted (may not have audio): {result.stderr[:200]}")
            return None

        # Verify the file was actually created and has content
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            logger.info(f"Audio extracted to {output_path}")
            return output_path
        return None

    except subprocess.TimeoutExpired:
        logger.warning("Audio extraction timed out.")
        return None
    except FileNotFoundError:
        logger.warning("ffmpeg not found for audio extraction.")
        return None


def process_all_frames(
    input_dir: str,
    output_dir: str,
    unmirror: bool = False,
    detint_strength: float = 0.9,
    clahe_clip_limit: float = 2.5,
    saturation_multiplier: float = 1.35,
    sharpen_amount: float = 1.2,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Process all extracted frames through the restoration pipeline.

    Parameters
    ----------
    input_dir : str
        Directory containing extracted PNG frames.
    output_dir : str
        Directory to save processed frames.
    unmirror : bool
        Whether to horizontally flip frames.
    detint_strength : float
        Color cast removal strength.
    clahe_clip_limit : float
        CLAHE contrast clip limit.
    saturation_multiplier : float
        Saturation boost factor.
    sharpen_amount : float
        Unsharp mask strength.
    progress_callback : callable, optional
        Called with (current_frame, total_frames) for progress reporting.

    Returns
    -------
    int
        Number of frames processed.
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(input_dir, "frame_*.png")))
    total = len(frame_files)

    if total == 0:
        raise RuntimeError(f"No frames found in {input_dir}")

    logger.info(f"Processing {total} frames...")

    for i, frame_path in enumerate(frame_files, 1):
        frame = cv2.imread(frame_path)
        if frame is None:
            logger.warning(f"Could not read frame: {frame_path}, skipping.")
            continue

        processed = process_frame(
            frame,
            unmirror=unmirror,
            detint_strength=detint_strength,
            clahe_clip_limit=clahe_clip_limit,
            saturation_multiplier=saturation_multiplier,
            sharpen_amount=sharpen_amount,
        )

        output_path = os.path.join(output_dir, os.path.basename(frame_path))
        cv2.imwrite(output_path, processed)

        if progress_callback:
            progress_callback(i, total)
        elif i % 10 == 0 or i == total:
            logger.info(f"Processing frame {i}/{total}")

    return total


def reassemble_video(
    frames_dir: str,
    audio_path: Optional[str],
    output_path: str,
    framerate: float,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> str:
    """
    Stitch processed frames back into a video, optionally remuxing audio.
    
    Ensures output dimensions match original video (padded to even if needed
    for yuv420p compatibility).

    Returns the output file path.
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    frame_pattern = os.path.join(frames_dir, "frame_%06d.png")

    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(framerate),
        "-i", frame_pattern,
    ]

    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-i", audio_path, "-c:a", "aac", "-shortest"])

    # Use scale filter to ensure even dimensions (yuv420p requires this)
    # -2 means "round to nearest even number"
    if width and height:
        # Pad to even dimensions if needed
        even_w = width if width % 2 == 0 else width + 1
        even_h = height if height % 2 == 0 else height + 1
        cmd.extend(["-vf", f"scale={even_w}:{even_h}:flags=lanczos"])
    else:
        # If we don't know the original size, just pad to even
        cmd.extend(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])

    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "18",
        output_path,
    ])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg video reassembly failed:\n{result.stderr}"
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError("Video reassembly timed out (>10 minutes).")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg. See SETUP.md."
        )

    logger.info(f"Reassembled video saved to {output_path}")
    return output_path


def cleanup_temp_files(*dirs: str) -> None:
    """Remove temporary directories and their contents."""
    for d in dirs:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
                logger.info(f"Cleaned up: {d}")
            except OSError as e:
                logger.warning(f"Could not fully clean {d}: {e}")


def restore_video(
    input_path: str,
    output_path: str,
    unmirror: bool = False,
    detint_strength: float = 0.9,
    clahe_clip_limit: float = 2.5,
    saturation_multiplier: float = 1.35,
    sharpen_amount: float = 1.2,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> str:
    """
    Full restoration pipeline: extract → process → reassemble → cleanup.

    Parameters
    ----------
    input_path : str
        Path to the input video file.
    output_path : str
        Path for the restored output video.
    unmirror : bool
        Whether to un-mirror (horizontally flip) frames.
    detint_strength : float
        Color cast removal strength (0.0–1.0).
    clahe_clip_limit : float
        CLAHE contrast clip limit.
    saturation_multiplier : float
        Saturation boost multiplier.
    sharpen_amount : float
        Unsharp mask strength.
    progress_callback : callable, optional
        Progress reporting function(current, total).

    Returns
    -------
    str
        Path to the restored video file.

    Raises
    ------
    FileNotFoundError
        If input file doesn't exist.
    ValueError
        If input format is not supported.
    RuntimeError
        If ffmpeg operations fail.
    """
    # Validate input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported video format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_FORMATS)}"
        )

    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg is not installed or not found in PATH. "
            "Please install ffmpeg. See SETUP.md for instructions."
        )

    # Set up temp directories (use unique names to avoid collisions)
    base_tmp = os.path.join(os.path.dirname(input_path), "_frames_tmp")
    raw_frames_dir = os.path.join(base_tmp, "raw")
    processed_frames_dir = os.path.join(base_tmp, "processed")
    audio_path = os.path.join(base_tmp, "audio.aac")

    try:
        # Step 1: Extract frames
        logger.info("Step 1/4: Extracting frames...")
        video_info = extract_frames(input_path, raw_frames_dir)
        framerate = video_info["framerate"]

        # Step 2: Extract audio
        logger.info("Step 2/4: Extracting audio...")
        extracted_audio = extract_audio(input_path, audio_path)

        # Step 3: Process all frames
        logger.info("Step 3/4: Processing frames...")
        process_all_frames(
            raw_frames_dir,
            processed_frames_dir,
            unmirror=unmirror,
            detint_strength=detint_strength,
            clahe_clip_limit=clahe_clip_limit,
            saturation_multiplier=saturation_multiplier,
            sharpen_amount=sharpen_amount,
            progress_callback=progress_callback,
        )

        # Step 4: Reassemble video
        logger.info("Step 4/4: Reassembling video...")
        reassemble_video(
            processed_frames_dir, extracted_audio, output_path, framerate,
            width=video_info.get("width"),
            height=video_info.get("height"),
        )

        logger.info(f"Restoration complete: {output_path}")
        return output_path

    finally:
        # Always clean up temp files
        cleanup_temp_files(base_tmp)
