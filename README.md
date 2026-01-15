# 🎬 YouTube Automation Bot - Self-Learning AI Video Enhancement

**Transform reused content into viral-ready videos with Hybrid Vision AI, Self-Learning Watermark Detection, and Smart Audio Remixing.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## ✨ Key Features

### 🧠 **Self-Learning Watermark System (Triple Refinement)**

- **Hybrid Vision Engine**: Combines **Google Gemini Vision** (Contextual Detection) + **OpenCV ORB** (Precise Tracking) + **Machine Learning** (Validation).
- **Tri-Level Removal**:
  - **Static Delogo**: For fixed logos (with specific time-range support).
  - **Dynamic Masking**: For moving watermarks using trajectory tracking and clamp logic.
  - **Drift Clamp**: Prevents mask from wandering off the watermark during motion.
- **Feedback Loop**: Learns from your Telegram commands:
  - `Yes` / `/approve`: "Correct detection." (Reinforces Positive Model)
  - `No` / `/reject`: "Wrong detection." (Reinforces Negative Model & Triggers **Aggressive Retry**)

### 🗣️ **AI Narrator & Voiceover (NEW)**

- **AI Voiceover**: Automatically generates voiceovers from AI-generated captions or Gemini summaries.
- **Smart Mixing**: Auto-ducks background music to ensure the voiceover is crisp and clear.

### 🎨 **Transformative Content Engine**

- **Gemini AI Captions**: Generates viral-style captions, titles, and hashtags using Gemini 1.5 Flash.
- **Smart Cropping**: Intelligent content-aware cropping for 9:16 vertical format.
- **Dynamic Text Overlays**: Professional-grade text rendering with shadows and borders.
- **Cinematic Color Grading**: Applies "Dark", "Vibrant", "Cinematic", "Warm", or "Cool" LUTs.
- **Speed Ramping**: Dynamic speed variations (±15%) to avoid copyright matching.

### 🎵 **Advanced Audio Studio**

- **Heavy Remixing**: Completely transforms audio structure using beat-aware slicing for Shorts and Compilations.
- **Auto-Generated Music**: Creates unique, copyright-free background music on the fly if no suitable tracks are found.
- **Continuous Mixes**: Stitches multiple tracks from your library for long compilations.

### 🚀 **Performance & Architecture**

- **Smart Hardware Detection**: Auto-switches between NVENC (GPU) and libx264 (CPU).
- **Batch Compilation**: Merges multiple processed clips with smart transitions (Fade, Wipe, Zoom, Slide).
- **Incremental Renaming**: Ensures no files are overwritten (`_1`, `_2`, etc.).

---

## 🚀 Quick Start

### **Option 1: Google Colab (Recommended for GPU)**

1. **Clone the repository:**

   ```bash
   !git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   %cd YOUR_REPO
   ```

2. **Run installation:**

   ```bash
   !python install_colab.py
   ```

3. **Start the bot:**
   ```bash
   !python main.py
   ```

---

## ⚙️ Configuration (`.env`)

### **Core Settings**

```ini
TELEGRAM_BOT_TOKEN=your_token
GEMINI_API_KEY=your_key
```

### **Watermark & Vision**

```ini
WATERMARK_DETECTION=yes
ENABLE_HYBRID_VISION=yes
WATERMARK_DETECT_LIST=tiktok,reels,watermark,@  # Keywords for Gemini
```

### **Audio & Voiceover**

```ini
ENABLE_MICRO_VOICEOVER=yes      # Enable AI Voiceover
ENABLE_HEAVY_REMIX_SHORTS=yes   # Aggressive remixing
ENABLE_AUTO_MUSIC_GEN=yes       # Generate music if needed
```

---

## 🎬 Usage

### **Telegram Commands**

| Command              | Description                                |
| :------------------- | :----------------------------------------- |
| `/start`             | Start the bot and get instructions.        |
| `/setbatch <N>`      | Set auto-compilation threshold (e.g., 6).  |
| `/getbatch`          | View current batch size.                   |
| `/compile_last <N>`  | Compile the **latest** N processed videos. |
| `/compile_first <N>` | Compile the **oldest** N processed videos. |
| `/approve`           | Confirm upload to YouTube.                 |
| `/reject`            | Discard video and delete file.             |

### **Workflow**

1. **Send Video**: Bot downloads (`downloader.py`) and analyzes metadata.
2. **Processing**:
   - **Enhancement**: Upscales and normalizes to 1080x1920.
   - **Watermark**: Hybrid Vision detects and removes watermarks (Static or Dynamic).
   - **Transform**: Applies Color Grading, Speed Ramping, and Text Overlays.
   - **Audio**: Generates Voiceover (if enabled) and Remixes Audio.
3. **Review**: Bot sends a preview to Telegram.
   - _Message:_ "(Watermark detected - please verify removal- yes/no)"
4. **Feedback**:
   - Reply **`Yes`**: Confirms valid detection (Learns Positive).
   - Reply **`No`**: Reports miss/error. Bot **automatically retries** with **Aggressive Mode**.
5. **Finalize**:
   - Reply **`/approve`**: Uploads to YouTube (if configured).
   - Reply **`/reject`**: Deletes the video.

---

## 📁 Project Structure

```
├── main.py                 # Bot entry point & Telegram handlers
├── compiler.py             # Core video processing pipeline (The Brain)
├── watermark_auto.py       # Hybrid Vision orchestration
├── hybrid_watermark.py     # Advanced Watermark Logic (Drift Clamp, Trajectory)
├── gemini_captions.py      # AI Captioning & Vision
├── voiceover.py            # AI Voiceover generation
├── audio_processing.py     # Remixing & Music Generation
├── ai_engine.py            # Upscaling (Real-ESRGAN/GFPGAN)
├── watermark_templates/    # Learned templates (Positive/Negative)
├── watermark_dataset.csv   # ML Training Data
└── requirements.txt        # Dependencies
```

---

## 📊 Performance

| Hardware              | Mode          | Video (23s) | Processing Time |
| :-------------------- | :------------ | :---------- | :-------------- |
| **Google Colab (T4)** | GPU (NVENC)   | 23s         | ~35s ✅         |
| **CPU (Standard)**    | CPU (libx264) | 23s         | ~2-3 mins       |

---

**Made with ❤️ for content creators**
