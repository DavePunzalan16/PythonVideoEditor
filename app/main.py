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
from app.pipeline.face_restore import is_face_restore_available

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
    Upload a filtered/posterized video and this tool will clean it up — 
    removing color casts, reducing blockiness, and restoring contrast.
    
    > ⚠️ This produces a *cleaned approximation*. Detail lost to heavy 
    > posterization cannot be perfectly recovered.
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
if "original_filename" not in st.session_state:
    st.session_state.original_filename = None

# ─── Sidebar Controls ────────────────────────────────────────────────────────

st.sidebar.header("⚙️ Processing Parameters")
st.sidebar.markdown("Adjust these to fine-tune the restoration.")

unmirror = st.sidebar.checkbox(
    "🔄 Un-mirror (flip horizontally)",
    value=False,
    help="Check if text appears backwards in the video.",
)

detint_strength = st.sidebar.slider(
    "De-tint Strength",
    min_value=0.0,
    max_value=1.0,
    value=0.85,
    step=0.05,
    help="How aggressively to remove green/yellow color cast. 1.0 = full removal.",
)

saturation_boost = st.sidebar.slider(
    "Saturation Boost",
    min_value=0.5,
    max_value=2.5,
    value=1.3,
    step=0.05,
    help="Color vibrancy multiplier. Higher = more vivid.",
)

contrast_strength = st.sidebar.slider(
    "Contrast Strength (CLAHE)",
    min_value=0.5,
    max_value=5.0,
    value=2.0,
    step=0.25,
    help="Adaptive contrast intensity. Higher = stronger contrast.",
)

sharpen_amount = st.sidebar.slider(
    "Sharpening Amount",
    min_value=0.0,
    max_value=2.0,
    value=0.8,
    step=0.1,
    help="Edge sharpening. Keep low to avoid artifacts on faces.",
)

st.sidebar.markdown("---")

# ─── AI Face Restoration ─────────────────────────────────────────────────────

st.sidebar.header("🧠 AI Face Restoration")

face_restore_enabled = st.sidebar.checkbox(
    "Generate realistic face (AI reconstruction)",
    value=False,
    help="Uses GFPGAN to reconstruct facial features from posterized blobs.",
)

if face_restore_enabled:
    if not is_face_restore_available():
        st.sidebar.warning(
            "⚠️ GFPGAN not installed. Face restoration won't work. "
            "Install: `pip install gfpgan torch torchvision basicsr facexlib`"
        )

    st.sidebar.markdown(
        """
        <div style="background-color: #2d2d2d; padding: 10px; border-radius: 5px; 
        font-size: 12px; border-left: 3px solid #ff9800;">
        ⚠️ <b>Disclaimer:</b> This generates a plausible, natural-looking 
        human face using AI. It is a <b>reconstruction</b>, not a guaranteed 
        match to the real person's exact features. This tool does not accept 
        reference photos of specific people.
        </div>
        """,
        unsafe_allow_html=True,
    )

    face_blend = st.sidebar.slider(
        "Face Restoration Strength",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="0.0 = original only, 1.0 = full AI reconstruction. 0.7 is a good balance.",
    )

    st.sidebar.caption("⏱️ Adds significant processing time. GPU recommended.")
else:
    face_blend = 0.7

st.sidebar.markdown("---")

# ─── Reset Button ────────────────────────────────────────────────────────────

if st.sidebar.button("🔄 Reset / New Video"):
    st.session_state.restored_path = None
    st.session_state.original_path = None
    st.session_state.processing_done = False
    st.session_state.original_filename = None
    st.rerun()

# ─── Show Completed Result (if already processed) ────────────────────────────

if st.session_state.processing_done and st.session_state.restored_path:
    restored_path = st.session_state.restored_path
    original_path = st.session_state.original_path
    original_filename = st.session_state.original_filename or "video.mp4"

    # ── Completed Section ──
    st.success("✅ Video restoration complete!")

    st.subheader("📺 Restored Result")
    if os.path.exists(restored_path):
        st.video(restored_path)
    else:
        st.warning("Restored file no longer exists. Please process again.")

    # Download buttons
    st.markdown("### ⬇️ Downloads")
    col1, col2 = st.columns(2)

    with col1:
        if os.path.exists(restored_path):
            with open(restored_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Restored Video",
                    data=f.read(),
                    file_name=f"restored_{original_filename}",
                    mime="video/mp4",
                    type="primary",
                )

    with col2:
        if original_path and os.path.exists(original_path):
            with open(original_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Original Upload",
                    data=f.read(),
                    file_name=original_filename,
                    mime="video/mp4",
                )

    # Before / After comparison
    st.markdown("---")
    st.subheader("🔍 Before & After Comparison")
    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("**Original (filtered)**")
        if original_path and os.path.exists(original_path):
            st.video(original_path)

    with col_after:
        st.markdown("**Restored**")
        if os.path.exists(restored_path):
            st.video(restored_path)

    st.info(
        "💡 Not happy with the result? Adjust the sliders in the sidebar "
        "and click **Reset** to try again with different settings."
    )

    st.stop()  # Don't show the upload section when results are displayed

# ─── File Upload ──────────────────────────────────────────────────────────────

st.markdown("---")
uploaded_file = st.file_uploader(
    "Upload a filtered video",
    type=SUPPORTED_TYPES,
    help=f"Supported: {', '.join(SUPPORTED_TYPES)}. Max: {MAX_FILE_SIZE_MB}MB.",
)

if uploaded_file is not None:
    # Size validation
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(
            f"❌ File too large ({file_size_mb:.1f}MB). "
            f"Max supported: {MAX_FILE_SIZE_MB}MB. "
            f"Please trim your video first."
        )
        st.stop()

    # Save uploaded file to temp location
    tmp_dir = tempfile.mkdtemp(prefix="filter_restore_")
    original_filename = uploaded_file.name
    input_path = os.path.join(tmp_dir, original_filename)

    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.session_state.original_path = input_path
    st.session_state.original_filename = original_filename

    # Preview original
    st.subheader("📹 Uploaded Video Preview")
    st.video(input_path)
    st.caption(f"{original_filename} — {file_size_mb:.1f}MB")

    # ─── Process Button ───────────────────────────────────────────────────

    if st.button("🚀 Restore Video", type="primary"):
        output_filename = f"restored_{original_filename}"
        # Ensure output is .mp4 for browser compatibility
        if not output_filename.endswith(".mp4"):
            output_filename = os.path.splitext(output_filename)[0] + ".mp4"
        output_path = os.path.join(tmp_dir, output_filename)

        progress_bar = st.progress(0, text="Starting restoration...")

        def update_progress(current: int, total: int):
            pct = current / total
            progress_bar.progress(pct, text=f"Processing frame {current}/{total}")

        try:
            restore_video(
                input_path=input_path,
                output_path=output_path,
                unmirror=unmirror,
                detint_strength=detint_strength,
                clahe_clip_limit=contrast_strength,
                saturation_multiplier=saturation_boost,
                sharpen_amount=sharpen_amount,
                face_restore=face_restore_enabled,
                face_blend_weight=face_blend,
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
