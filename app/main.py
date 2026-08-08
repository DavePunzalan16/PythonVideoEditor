"""
Streamlit entry point for the Video Filter Restoration app.

Run with: streamlit run app/main.py
"""

import os
import sys
import tempfile
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from app.pipeline.video_pipeline import restore_video, check_ffmpeg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("restore_app.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB = 200
SUPPORTED_TYPES = ["mp4", "mov", "avi"]

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Video Filter Restoration",
    page_icon="🎬",
    layout="wide",
)

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🎬 Video Filter Restoration")
st.markdown(
    """
    **Upload a filtered/posterized video** and this tool will restore it as close 
    to natural as possible — removing color casts, reducing blocky posterization, 
    and boosting contrast and clarity.
    
    ⚠️ **Important:** This produces a *cleaned approximation*, not a perfect reversal. 
    Detail permanently lost to heavy posterization or beauty-smoothing filters 
    cannot be mathematically recovered — only approximated.
    """
)

# ─── FFmpeg Check ─────────────────────────────────────────────────────────────

if not check_ffmpeg():
    st.error(
        "⚠️ **ffmpeg not found on this system.** "
        "This app requires ffmpeg to process video files. "
        "Please see **SETUP.md** for installation instructions."
    )
    st.stop()

# ─── Session State Init ──────────────────────────────────────────────────────

if "restored_path" not in st.session_state:
    st.session_state.restored_path = None
if "original_path" not in st.session_state:
    st.session_state.original_path = None
if "processing_done" not in st.session_state:
    st.session_state.processing_done = False

# ─── Sidebar Controls ────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Processing Parameters")
st.sidebar.markdown("Adjust these to fine-tune restoration for your specific video.")

unmirror = st.sidebar.checkbox(
    "🔄 Video appears mirrored/flipped — un-mirror it",
    value=False,
    help="Check this if text in the video appears backwards or the orientation is flipped.",
)

detint_strength = st.sidebar.slider(
    "De-tint Strength",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.05,
    help="How aggressively to remove green/yellow color cast. 1.0 = full removal.",
)

saturation_boost = st.sidebar.slider(
    "Saturation Boost",
    min_value=0.5,
    max_value=2.5,
    value=1.35,
    step=0.05,
    help="Multiplier for color saturation. Higher = more vivid colors.",
)

contrast_strength = st.sidebar.slider(
    "Contrast Strength (CLAHE)",
    min_value=0.5,
    max_value=5.0,
    value=2.5,
    step=0.25,
    help="Adaptive contrast clip limit. Higher = stronger contrast restoration.",
)

sharpen_amount = st.sidebar.slider(
    "Sharpening Amount",
    min_value=0.0,
    max_value=3.0,
    value=1.2,
    step=0.1,
    help="Unsharp mask strength. Higher = sharper but may introduce artifacts.",
)

# ─── Reset Button ────────────────────────────────────────────────────────────

if st.sidebar.button("🔄 Reset / Process Another Video"):
    st.session_state.restored_path = None
    st.session_state.original_path = None
    st.session_state.processing_done = False
    st.rerun()

# ─── File Upload ──────────────────────────────────────────────────────────────

st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload a filtered video",
    type=SUPPORTED_TYPES,
    help=f"Supported formats: {', '.join(SUPPORTED_TYPES)}. Max size: {MAX_FILE_SIZE_MB}MB.",
)

if uploaded_file is not None:
    # Size validation
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"❌ File too large ({file_size_mb:.1f}MB). "
            f"Maximum supported size is {MAX_FILE_SIZE_MB}MB. "
            f"Frame-by-frame processing is very slow on large files — "
            f"please trim your video first."
        )
        st.stop()

    # Save uploaded file to temp location
    tmp_dir = tempfile.mkdtemp(prefix="filter_restore_")
    original_filename = uploaded_file.name
    input_path = os.path.join(tmp_dir, original_filename)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.original_path = input_path

    # Preview original
    st.subheader("📹 Original Upload")
    st.video(input_path)
    st.caption(f"File: {original_filename} ({file_size_mb:.1f}MB)")

    # ─── Process Button ───────────────────────────────────────────────────

    if not st.session_state.processing_done:
        if st.button("🚀 Restore Video", type="primary"):
            output_filename = f"restored_{original_filename}"
            output_path = os.path.join(tmp_dir, output_filename)

            progress_bar = st.progress(0, text="Starting restoration...")
            status_text = st.empty()

            def update_progress(current: int, total: int):
                pct = current / total
                progress_bar.progress(pct, text=f"Processing frame {current}/{total}")

            try:
                with st.spinner("Processing video... This may take a while for longer clips."):
                    restore_video(
                        input_path=input_path,
                        output_path=output_path,
                        unmirror=unmirror,
                        detint_strength=detint_strength,
                        clahe_clip_limit=contrast_strength,
                        saturation_multiplier=saturation_boost,
                        sharpen_amount=sharpen_amount,
                        progress_callback=update_progress,
                    )

                st.session_state.restored_path = output_path
                st.session_state.processing_done = True
                progress_bar.progress(1.0, text="✅ Done!")
                st.rerun()

            except RuntimeError as e:
                st.error(f"❌ Processing failed: {e}")
                logger.error(f"Processing failed: {e}", exc_info=True)
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")
                logger.error(f"Unexpected error: {e}", exc_info=True)

    # ─── Results Display ──────────────────────────────────────────────────

    if st.session_state.processing_done and st.session_state.restored_path:
        restored_path = st.session_state.restored_path

        st.markdown("---")
        st.subheader("✅ Restored Video")
        st.video(restored_path)

        # Download buttons side by side
        col1, col2 = st.columns(2)

        with col1:
            if os.path.exists(restored_path):
                with open(restored_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Restored Video",
                        data=f.read(),
                        file_name=f"restored_{original_filename}",
                        mime="video/mp4",
                    )

        with col2:
            if os.path.exists(input_path):
                with open(input_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Original Upload",
                        data=f.read(),
                        file_name=original_filename,
                        mime="video/mp4",
                    )

        st.info(
            "💡 **Tip:** Compare the original and restored versions side by side. "
            "If the result still looks too green or too flat, adjust the sliders "
            "in the sidebar and process again."
        )
