"""
End-to-end test: creates a synthetic filtered video, runs the full
restoration pipeline, and verifies the output is a valid video file.
"""

import os
import sys
import tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from app.pipeline.video_pipeline import restore_video, check_ffmpeg


def create_test_video(output_path: str, frames: int = 30, fps: int = 15):
    """Create a short synthetic green-tinted posterized video."""
    width, height = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(frames):
        # Create posterized green-tinted frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Moving gradient with green tint
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)

        # Posterized blocks
        block_size = 20
        bx = (xx // block_size) * block_size
        by = (yy // block_size) * block_size

        frame[:, :, 0] = ((bx + i * 3) % 80 + 20).astype(np.uint8)   # B: low
        frame[:, :, 1] = ((by + bx + i * 2) % 120 + 100).astype(np.uint8)  # G: high (green tint)
        frame[:, :, 2] = ((bx * 2 + i) % 60 + 30).astype(np.uint8)    # R: low

        writer.write(frame)

    writer.release()
    print(f"Created test video: {output_path} ({frames} frames, {fps}fps)")


def main():
    if not check_ffmpeg():
        print("ERROR: ffmpeg not found. Cannot run full pipeline test.")
        print("See SETUP.md for installation instructions.")
        sys.exit(1)

    # Create temp directory for test
    tmp_dir = tempfile.mkdtemp(prefix="pipeline_test_")
    input_path = os.path.join(tmp_dir, "test_input.mp4")
    output_path = os.path.join(tmp_dir, "test_restored.mp4")

    try:
        # Create test video
        create_test_video(input_path)

        # Run full pipeline
        print("\nRunning full restoration pipeline...")

        def progress(current, total):
            if current % 10 == 0 or current == total:
                print(f"  Frame {current}/{total}")

        result = restore_video(
            input_path=input_path,
            output_path=output_path,
            unmirror=False,
            progress_callback=progress,
        )

        # Verify output
        if os.path.exists(result) and os.path.getsize(result) > 0:
            size_kb = os.path.getsize(result) / 1024
            print(f"\n✅ SUCCESS! Output video: {result} ({size_kb:.1f} KB)")

            # Verify it's a valid video by opening with cv2
            cap = cv2.VideoCapture(result)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            print(f"   Output has {frame_count} frames.")
        else:
            print("\n❌ FAILED: Output file is missing or empty.")
            sys.exit(1)

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("\nTest cleanup done.")


if __name__ == "__main__":
    main()
