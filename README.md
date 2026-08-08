# Video Filter Restoration App

A web-based tool that restores heavily filtered/posterized video — removing green color casts, reducing blocky posterization artifacts, and restoring contrast and clarity.

## What It Does

This app takes video that has been processed with heavy "beauty" or artistic filters (green/gold tints, posterization, flattened contrast, mirror-flipping) and applies a multi-step correction pipeline:

1. **De-posterization** — Bilateral filtering smooths out blocky color regions
2. **De-tinting** — LAB color space correction neutralizes green/yellow color casts
3. **Contrast restoration** — CLAHE adaptive histogram equalization restores dynamic range
4. **Saturation boost** — HSV channel manipulation brings back natural color vibrancy
5. **Sharpening** — Unsharp masking restores perceived detail and edges
6. **Optional AI upscale** — Real-ESRGAN adds perceived HD detail (GPU required)

## Honest Limitations

- **This is restoration, not reversal.** Detail destroyed by heavy posterization or beauty-smoothing cannot be mathematically recovered. The pipeline produces a cleaned, corrected *approximation*.
- **Results vary by source.** Different filter apps/effects need different correction strengths — use the sliders to tune per-video.
- **Processing is slow.** Frame-by-frame processing takes time, especially for longer clips. Keep uploads short (under 30 seconds) for reasonable wait times.
- **AI upscale needs a GPU.** The optional Real-ESRGAN step is impractical on CPU.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure ffmpeg is installed (see SETUP.md for platform-specific instructions)
ffmpeg -version

# Run the app
streamlit run app/main.py
```

See [SETUP.md](SETUP.md) for detailed installation instructions.

## Project Structure

```
filter-restore-app/
├── app/
│   ├── main.py                 # Streamlit UI entry point
│   ├── pipeline/
│   │   ├── frame_processor.py  # Single-frame restoration logic
│   │   ├── video_pipeline.py   # Full video processing orchestration
│   │   └── upscale.py          # Optional AI upscaling (Real-ESRGAN)
│   └── utils/
├── output/                     # Test output directory
├── requirements.txt
├── test_single_image.py        # Quick single-frame test script
├── SETUP.md                    # Full installation guide
└── README.md
```

## Usage

1. Upload a filtered video (mp4, mov, or avi)
2. Adjust the sidebar sliders if needed:
   - **De-tint Strength**: How aggressively to remove color cast
   - **Saturation Boost**: Color vibrancy multiplier
   - **Contrast Strength**: Adaptive contrast intensity
   - **Sharpening Amount**: Edge sharpening level
3. Toggle "Un-mirror" if the video appears horizontally flipped
4. Click "Restore Video" and wait for processing
5. Preview the result and download both the restored and original files

## Deployment

This app can be deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) for free. Keep in mind:
- Processing time limits on free tiers may require keeping clips short (under 10 seconds)
- ffmpeg must be available in the deployment environment
- AI upscaling is not practical on free hosting (no GPU)

## License

MIT
