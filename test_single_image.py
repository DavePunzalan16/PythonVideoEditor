"""
Quick test script: loads a sample image, runs process_frame, saves the result.

Usage:
    python test_single_image.py [input_path] [output_path]

If no arguments are provided, creates a synthetic test image to verify
the pipeline runs without errors.
"""

import sys
import os
import numpy as np

# Add project root to path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
from app.pipeline.frame_processor import process_frame


def create_synthetic_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """
    Create a synthetic image that simulates a green-tinted posterized frame.
    Useful for testing when no real sample is available.
    """
    # Create a gradient base
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some structure (rectangles simulating posterized regions)
    for i in range(0, height, 60):
        for j in range(0, width, 80):
            # Posterized blocky greens/yellows
            g = int(80 + 100 * ((i + j) % 3) / 2)
            b = int(30 + 40 * (i % 2))
            r = int(40 + 60 * (j % 2))
            cv2.rectangle(img, (j, i), (j + 80, i + 60), (b, g, r), -1)

    # Apply a strong green tint
    img[:, :, 1] = np.clip(img[:, :, 1].astype(np.int16) + 60, 0, 255).astype(np.uint8)

    # Reduce contrast (flatten)
    img = (img.astype(np.float32) * 0.6 + 50).clip(0, 255).astype(np.uint8)

    return img


def main():
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
        if not os.path.exists(input_path):
            print(f"Error: Input file '{input_path}' not found.")
            sys.exit(1)
        frame = cv2.imread(input_path)
        if frame is None:
            print(f"Error: Could not read image '{input_path}'.")
            sys.exit(1)
    else:
        print("No input image provided. Creating synthetic test image...")
        frame = create_synthetic_test_image()

    output_path = sys.argv[2] if len(sys.argv) >= 3 else "output/test_result.png"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else "output", exist_ok=True)

    print(f"Input shape: {frame.shape}")
    print("Running process_frame...")

    result = process_frame(frame, unmirror=False)

    print(f"Output shape: {result.shape}")

    # Save input alongside output for comparison
    input_save_path = output_path.replace("_result", "_input").replace(".png", "_input.png")
    cv2.imwrite(input_save_path, frame)
    cv2.imwrite(output_path, result)

    print(f"Input saved to: {input_save_path}")
    print(f"Result saved to: {output_path}")
    print("Done! Compare the two files to see the restoration effect.")


if __name__ == "__main__":
    main()
