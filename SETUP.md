# Setup Instructions

This guide covers installing all dependencies to run the Video Filter Restoration app locally.

## Prerequisites

- **Python 3.10+** (tested with 3.13)
- **ffmpeg** (system-level, not just the Python wrapper)

---

## 1. Install Python Dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

---

## 2. Install ffmpeg

The app requires the `ffmpeg` and `ffprobe` binaries on your system PATH.

### Windows

**Option A: winget (recommended)**
```powershell
winget install Gyan.FFmpeg
```

**Option B: Chocolatey**
```powershell
choco install ffmpeg
```

**Option C: Manual download**
1. Download from https://www.gyan.dev/ffmpeg/builds/ (get the "essentials" build)
2. Extract the zip
3. Add the `bin` folder to your system PATH

### macOS

```bash
brew install ffmpeg
```

### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

### Linux (Fedora)

```bash
sudo dnf install ffmpeg
```

### Verify installation

```bash
ffmpeg -version
ffprobe -version
```

Both commands should print version info without errors.

---

## 3. Run the App

```bash
cd filter-restore-app
streamlit run app/main.py
```

The app will open in your browser at http://localhost:8501.

---

## 4. Optional: AI Upscaling (GPU recommended)

For the HD upscale feature, install additional dependencies:

```bash
pip install realesrgan torch torchvision
```

**Note:** This requires a CUDA-capable GPU for practical speed. CPU inference is extremely slow (minutes per frame). The base restoration pipeline works fine without this.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ffmpeg not found` | Ensure ffmpeg is in your PATH. Restart terminal after install. |
| `ModuleNotFoundError` | Make sure your venv is activated and requirements are installed. |
| Very slow processing | Normal for longer videos. Use short clips (5-30 seconds) for testing. |
| Out of memory | Reduce video resolution before uploading, or process shorter clips. |
| Permission errors on Windows | Run terminal as administrator, or use a different directory for the venv. |
