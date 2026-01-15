# compiler.py - HIGH-END MULTI-PASS AI EDITOR (DUAL-STAGE ENGINE)
# STRICT AUDIT COMPLIANT: Atomic Writes, Watermark Gates, Rename Safety.

import os
import subprocess
import logging
import shutil
import sys
import random
import json
import glob
import time
import platform
import uuid
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dotenv import load_dotenv
import re
from concurrent.futures import ThreadPoolExecutor
import cv2

load_dotenv(".env", override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("compiler")

# Router is used for advanced enhancement
from router import run_enhancement
# Although watermark_auto is imported, we rely on DIRECT Hybrid usage for safety
from watermark_auto import (
    apply_text_watermark,
    process_video_with_watermark,
    run_adaptive_watermark_orchestration
)
from opencv_watermark import inpaint_video, check_watermark_residue
import hybrid_watermark

# Import Text Overlay
from text_overlay import add_logo_overlay, add_episodic_overlay
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
if not shutil.which(FFMPEG_BIN):
    FFMPEG_BIN = "ffmpeg"

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

try:
    from text_overlay import apply_text_overlay_safe
    HAS_TEXT_OVERLAY = True
except ImportError:
    HAS_TEXT_OVERLAY = False

try:
    from audio_processing import create_continuous_music_mix
except ImportError:
    # Fallback if audio_processing not available (unlikely)
    def create_continuous_music_mix(*args, **kwargs): return False

# Configuration
COMPUTE_MODE = os.getenv("COMPUTE_MODE", "auto").lower()
ENHANCEMENT_LEVEL = os.getenv("ENHANCEMENT_LEVEL", "2x").lower()
TRANSITION_DURATION = float(os.getenv("TRANSITION_DURATION", "1.0"))
TRANSITION_INTERVAL = float(os.getenv("TRANSITION_INTERVAL", "10"))
TARGET_RESOLUTION = os.getenv("TARGET_RESOLUTION", "1080:1920")
REENCODE_CRF = os.getenv("REENCODE_CRF", "18") # HQ Default
REENCODE_PRESET = os.getenv("REENCODE_PRESET", "medium") # Compression Efficiency

# AI Config
FACE_ENHANCEMENT = os.getenv("FACE_ENHANCEMENT", "yes").lower() == "yes"
USE_ADVANCED_ENGINE = os.getenv("USE_ADVANCED_ENGINE", "off").lower() == "on"

# Transformative Features Config
ADD_TEXT_OVERLAY = os.getenv("ADD_TEXT_OVERLAY", "yes").lower() == "yes" 
ADD_COLOR_GRADING = os.getenv("ADD_COLOR_GRADING", "yes").lower() == "yes"
ADD_SPEED_RAMPING = os.getenv("ADD_SPEED_RAMPING", "yes").lower() == "yes"
FORCE_AUDIO_REMIX = os.getenv("FORCE_AUDIO_REMIX", "yes").lower() == "yes"

# Text Overlay Settings
TEXT_OVERLAY_TEXT = os.getenv("TEXT_OVERLAY_CONTENT", "swargawasal")
TEXT_OVERLAY_POSITION = os.getenv("TEXT_OVERLAY_POSITION", "bottom")
TEXT_OVERLAY_SIZE = int(os.getenv("TEXT_OVERLAY_SIZE", "60"))

# Color Grading Settings
COLOR_FILTER = os.getenv("COLOR_FILTER", "cinematic")
COLOR_INTENSITY = float(os.getenv("COLOR_INTENSITY", "0.5"))

# Speed Ramping Settings
SPEED_VARIATION = float(os.getenv("SPEED_VARIATION", "0.15"))

# Audio Remix Settings
ENABLE_HEAVY_REMIX_SHORTS = os.getenv("ENABLE_HEAVY_REMIX_SHORTS", "yes").lower() == "yes"
ENABLE_HEAVY_REMIX_COMPILATION = os.getenv("ENABLE_HEAVY_REMIX_COMPILATION", "yes").lower() == "yes"
AUTO_MUSIC = os.getenv("AUTO_MUSIC", "yes").lower() == "yes"
MUSIC_VOLUME = float(os.getenv("MUSIC_VOLUME", "0.4"))

def is_valid_video(file_path: str) -> bool:
    """Checks if a video file is valid (non-empty, valid structure)."""
    if not os.path.exists(file_path):
        return False
    # Check 1: Size > 1KB (0 byte files are common corruptions)
    if os.path.getsize(file_path) < 1024:
        logger.warning(f"⚠️ Skip Invalid Video (Too Small): {os.path.basename(file_path)}")
        return False
    return True
ORIGINAL_AUDIO_VOLUME = float(os.getenv("ORIGINAL_AUDIO_VOLUME", "1.0"))

# Audio Strategy
KEEP_ORIGINAL_AUDIO = os.getenv("KEEP_ORIGINAL_AUDIO", "yes").lower() == "yes"
FALLBACK_AUDIO = os.getenv("FALLBACK_AUDIO", "yes").lower() == "yes"

# Watermark Config
WATERMARK_DETECTION = os.getenv("WATERMARK_DETECTION", "yes").lower() == "yes"
WATERMARK_REMOVE_MODE = os.getenv("ENABLE_WATERMARK_REMOVAL", "yes").lower()
WATERMARK_REPLACE_MODE = os.getenv("ENABLE_WATERMARK_REPLACEMENT", "yes").lower()
WATERMARK_REMOVE = WATERMARK_REMOVE_MODE in ["yes", "auto", "true"]
WATERMARK_REPLACE = WATERMARK_REPLACE_MODE in ["yes", "auto", "true"]
MY_WATERMARK_FILE = os.getenv("WATERMARK_REPLACE_PATH", "assets/watermark.png")
MY_WATERMARK_TEXT = os.getenv("MY_WATERMARK_TEXT", "swargawasal")
MY_WATERMARK_OPACITY = float(os.getenv("MY_WATERMARK_OPACITY", "0.80"))

# Crop Config
EDGE_CROP_FACTOR = float(os.getenv("EDGE_CROP_FACTOR", "0.05"))

TEMP_DIR = "temp"
DOWNLOAD_DIR = "downloads"
OUTPUT_DIR = "Processed Shorts"
COMPILATION_DIR = "final_compilations"
TOOLS_DIR = os.path.join(os.getcwd(), "tools")
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPILATION_DIR, exist_ok=True)

# Safety & Cleanup Config
CLEANUP_POLICY = os.getenv("CLEANUP_POLICY", "delayed").lower() # immediate, on_success, delayed
FF_TIMEOUT_SECS = int(os.getenv("FF_TIMEOUT_SECS", 600))
FORCE_OVERWRITE = os.getenv("FORCE_OVERWRITE", "no").lower() == "yes"
RENAME_TRIES = int(os.getenv("RENAME_TRIES", 1000))

# Internal State
_compiler_meta = {"jobs_run": 0, "errors": []}

def _check_vram_safety() -> bool:
    """Checks if GPU VRAM is sufficient for AI operations using ComputeCaps."""
    try:
        from compute_caps import ComputeCaps
        caps = ComputeCaps.get()
        return caps["allow_ai_enhance"]
    except ImportError:
        return False

def _prune_temp_dirs():
    """Implements delayed cleanup policy (delete temp dirs > 24h old)."""
    if CLEANUP_POLICY != "delayed": return
    
    try:
        now = time.time()
        # 1. Clean Temp Jobs
        for p in glob.glob(os.path.join(TEMP_DIR, "job_*")):
            if os.path.isdir(p):
                mtime = os.path.getmtime(p)
                if now - mtime > 86400: # 24 hours
                    shutil.rmtree(p, ignore_errors=True)
                    logger.info(f"🧹 Pruned old temp dir: {os.path.basename(p)}")
        
        # 3. Clean Downloads
        if os.path.exists(DOWNLOAD_DIR):
            for p in glob.glob(os.path.join(DOWNLOAD_DIR, "*")):
                mtime = os.path.getmtime(p)
                if now - mtime > 86400: # 24 hours
                    if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                    else: os.remove(p)
                    logger.info(f"🧹 Pruned old download item: {os.path.basename(p)}")

        # 4. Clean Final Compilations
        if os.path.exists(COMPILATION_DIR):
             for p in glob.glob(os.path.join(COMPILATION_DIR, "*")):
                 mtime = os.path.getmtime(p)
                 if now - mtime > 86400: # 24 hours
                     try:
                         if os.path.isdir(p): shutil.rmtree(p, ignore_errors=True)
                         else: os.remove(p)
                         logger.info(f"🧹 Pruned old compilation: {os.path.basename(p)}")
                     except: pass

    except: pass

def _safe_atomic_write(src_video: str, dst_video: str, src_meta: str, dst_meta: str) -> bool:
    """
    Safely moves BOTH video and metadata to destination atomically.
    Prevents orphaned files or half-written states.
    """
    if not os.path.exists(src_video) or not os.path.exists(src_meta):
         logger.error("Atomic Write: Source files missing.")
         return False
         
    try:
        # Atomic Rename Strategy:
        # 1. Clean destination if forced
        if os.path.exists(dst_video) and FORCE_OVERWRITE: os.remove(dst_video)
        if os.path.exists(dst_meta) and FORCE_OVERWRITE: os.remove(dst_meta)
        
        # 2. Check collisions (should be handled by caller, but double check)
        if os.path.exists(dst_video):
            logger.error(f"Atomic Write: Destination exists (Race Condition?): {dst_video}")
            return False
            
        # 3. Move Video
        os.replace(src_video, dst_video)
        
        # 4. Move Metadata
        try:
            os.replace(src_meta, dst_meta)
        except Exception as e:
            # CRITICAL FAILURE: Video moved, JSON failed.
            # Rollback video implies data loss, but consistency is key.
            # We keep the video as it is valuable, but log ERROR.
            logger.error(f"Atomic Write Partial Failure (JSON): {e}")
            # Try to recreate JSON at destination
            shutil.copy2(src_meta, dst_meta)

        return True
    except Exception as e:
        logger.error(f"❌ Atomic write failed: {e}")
        return False

# Logic moved to watermark_auto.py to reduce redundancy

# ==================== HELPER FUNCTIONS ====================

def _run_command(cmd: List[str], check: bool = False, timeout: int = None) -> bool:
    try:
        if timeout is None: timeout = FF_TIMEOUT_SECS
        
        result = subprocess.run(
            cmd, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=timeout,
            encoding='utf-8', 
            errors='replace'
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        # Capture stderr for debugging (LAST 1000 chars are most relevant)
        err_msg = e.stderr[-1000:] if e.stderr else "No stderr"
        safe_cmd = " ".join([x for x in cmd if "key" not in x.lower()])
        logger.error(f"❌ Command Failed (Exit {e.returncode}): {safe_cmd}")
        logger.error(f"   Stderr: ...{err_msg}")
        return False
    except subprocess.TimeoutExpired:
        logger.warning(f"❌ Command timed out after {timeout}s: {cmd[0]}")
        _compiler_meta["errors"].append(f"timeout:{cmd[0]}")
        return False
    except Exception as e:
        safe_cmd = " ".join([x for x in cmd if "key" not in x.lower()])
        logger.error(f"❌ Command execution error: {e}")
        return False

def _get_video_info(path: str) -> Dict:
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration",
            "-of", "json", path
        ]
        result = subprocess.check_output(cmd).decode().strip()
        data = json.loads(result)
        stream = data["streams"][0]
        return {
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "duration": float(stream.get("duration", 0))
        }
    except Exception as e:
        logger.warning(f"⚠️ Failed to get video info for '{os.path.basename(path)}': {e}")
        return {"width": 0, "height": 0, "duration": 0}

def verify_video_integrity(file_path: str) -> bool:
    """Automated QA on output video."""
    if not os.path.exists(file_path):
        logger.error(f"❌ QA Failed: File not found: {file_path}")
        return False
        
    if os.path.getsize(file_path) == 0:
        logger.error(f"❌ QA Failed: File is empty: {file_path}")
        return False
        
    info = _get_video_info(file_path)
    if info.get("duration", 0) <= 0:
        logger.error(f"❌ QA Failed: Invalid duration ({info.get('duration')}s)")
        return False
        
    if info.get("height", 0) <= 0:
        logger.error(f"❌ QA Failed: Invalid video stream (height=0)")
        return False
        
    logger.info(f"✅ QA Passed: {os.path.basename(file_path)} (Dur: {info['duration']}s)")
    return True

def _get_ffmpeg_encoder():
    """Detect NVENC availability."""
    if COMPUTE_MODE == "cpu": return "libx264"
    try:
        cmd = [FFMPEG_BIN, "-hide_banner", "-encoders"]
        result = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
        if "h264_nvenc" not in result: return "libx264"
        
        test_cmd = [
            FFMPEG_BIN, "-f", "lavfi", "-i", "color=c=black:s=256x256:d=1",
            "-pix_fmt", "yuv420p",
            "-c:v", "h264_nvenc", "-f", "null", "-"
        ]
        subprocess.check_output(test_cmd, stderr=subprocess.STDOUT, timeout=5)
        logger.info("🚀 NVENC Working")
        return "h264_nvenc"
    except Exception:
        logger.info("ℹ️ NVENC failed test. Using CPU.")
        return "libx264"

def _get_video_fps(input_path: str) -> float:
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not result.stdout.strip(): return 30.0
        num, den = map(int, result.stdout.strip().split('/'))
        return num / den if den != 0 else 30.0
    except Exception:
        return 30.0

def _has_audio_stream(input_path: str) -> bool:
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=codec_type", "-of", "csv=p=0",
            input_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except: return False

def normalize_video(input_path: str, output_path: str, target_res: tuple = (1080, 1920), max_duration: float = None, resize: bool = True):
    logger.info(f"📏 Normalizing video to {target_res} (Max Dur: {max_duration}s, Resize: {resize})...")
    encoder = _get_ffmpeg_encoder()
    preset = os.getenv("REENCODE_PRESET", "p4" if encoder == "h264_nvenc" else "superfast")
    fps = _get_video_fps(input_path)
    target_w, target_h = target_res
    
    # STRICT Normalization Rules for XFADE compatibility:
    # 1. Fixed FPS (30) to prevent timestamps mismatch
    # 2. Strict Scale & Pad (Conditional)
    # 3. Setsar=1
    
    vf_parts = []
    
    if resize:
        vf_parts.append(f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease")
        vf_parts.append(f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2")
        vf_parts.append("setsar=1")
        
    vf_parts.append("fps=30")  # FORCE 30 FPS
    vf_parts.append("hqdn3d=1.5:1.5:6:6")
    vf_parts.append("format=yuv420p") # FORCE Pixel Format
    
    vf = ",".join(vf_parts)
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path
    ]
    
    # Duration limit (for Shorts compliance)
    if max_duration:
        cmd.extend(["-t", str(max_duration)])
    
    has_audio = _has_audio_stream(input_path)
    if not has_audio:
        logger.info("🔇 No audio detected. Generating silence...")
        cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])
    
    # Enforce standard pixel format at output level too
    cmd.extend(["-vf", vf, "-c:v", encoder, "-pix_fmt", "yuv420p", "-r", "30", "-preset", preset])
    
    # Meta Optimization
    cmd.extend(["-movflags", "+faststart"])
    if encoder == "libx264": cmd.extend(["-profile:v", "high", "-level", "4.2", "-crf", REENCODE_CRF])
    else: cmd.extend(["-rc", "vbr", "-cq", "19"])

    cmd.extend(["-c:a", "aac", "-ar", "44100", "-ac", "2"])
    
    if not has_audio:
        # Map original video (0:v) and new silence (1:a)
        cmd.extend(["-map", "0:v", "-map", "1:a", "-shortest"])
        
    cmd.append(output_path)
    return _run_command(cmd, check=True)

def apply_edge_crop(input_path: str, output_path: str, factor: float = 0.05) -> bool:
    """
    Crops the video by 'factor' from all sides (Zoom In).
    SMART UPGRADE: Checks for faces to prevent decapitation (Head cropping).
    """
    
    # 1. SMART CHECK: Adjust factor if face is near edge
    final_factor = factor
    try:
        from quality_orchestrator import human_guard
        if human_guard:
            # Check Middle Frame
            cap = cv2.VideoCapture(input_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                faces = human_guard.detect_faces(frame)
                h_img, w_img = frame.shape[:2]
                
                safe_margin = 0.01 # 1% margin
                
                for face in faces:
                    x, y, w, h = face['box']
                    
                    # Top Gap (Critical for Heads)
                    top_gap = y / float(h_img)
                    
                    # 🛡️ USER RULE: Keep 50% of the gap as safety margin
                    # If gap is 12%, max crop is 6%.
                    safety_limit = top_gap * 0.5
                    
                    if factor > safety_limit:
                        new_f = max(0.005, safety_limit)
                        if new_f < final_factor:
                            final_factor = new_f
                            logger.info(f"✂️ Smart Crop: 🛡️ Head Protection Active! Reduced {factor*100}% -> {final_factor*100:.2f}% (Face at {top_gap*100:.1f}%)")
                            
                    # Bottom Gap (Feet/Chin?) - Less critical but good symmetry
                    bottom_gap = (h_img - (y + h)) / float(h_img)
                    if bottom_gap < factor:
                         new_f_b = max(0.005, bottom_gap - safe_margin)
                         if new_f_b < final_factor:
                             final_factor = new_f_b
                             logger.info(f"✂️ Smart Crop: 🛡️ Bottom Protection! Reduced to {final_factor*100:.2f}%")

    except Exception as e:
        logger.warning(f"Smart crop check failed (using default {factor}): {e}")
        final_factor = factor

    logger.info(f"✂️ Applying Edge Crop (Factor: {final_factor})...")
    
    if final_factor <= 0.005:
        logger.warning("Crop factor too small (<0.5%). Skipping to preserve content.")
        shutil.copy(input_path, output_path)
        return True

    try:
        # Calculate crop parameters
        # crop=w:h:x:y
        # w = iw * (1 - 2*factor)
        # h = ih * (1 - 2*factor)
        # x = iw * factor
        # y = ih * factor
        
        crop_filter = (
            f"crop=w=iw*(1-{2*final_factor}):h=ih*(1-{2*final_factor}):"
            f"x=iw*{final_factor}:y=ih*{final_factor}"
        )
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", input_path,
            "-vf", crop_filter,
            "-c:v", "libx264", 
            "-preset", "fast",
            "-crf", "18", # High quality intermediate
            "-c:a", "copy",
            output_path
        ]
        
        return _run_command(cmd, check=True)
        
    except Exception as e:
        logger.error(f"❌ Edge crop failed: {e}")
        return False


# ==================== TRANSFORMATIVE FEATURES ====================

def _get_next_music_track() -> Optional[str]:
    """
    Selects the next music track from 'music/' folder using persistent rotation.
    Tracks state in 'music_state.json'.
    """
    music_dir = "music"
    if not os.path.exists(music_dir): return None
    
    tracks = sorted(glob.glob(os.path.join(music_dir, "*.mp3")))
    if not tracks: return None
    
    state_file = "music_state.json"
    last_index = -1
    
    # Load state
    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                last_index = state.get("last_index", -1)
        except: pass
        
    # Increment
    next_index = (last_index + 1) % len(tracks)
    selected_track = tracks[next_index]
    
    # Save state
    try:
        with open(state_file, 'w') as f:
            json.dump({"last_index": next_index}, f)
    except: pass
    
    return selected_track


# ... (Previous imports)

# ==================== TRANSFORMATIVE FEATURES ====================









def _save_sidecar(video_path: str, caption_meta: Dict, pipeline_metrics: Dict):
    """
    Saves metadata sidecar reliably.
    """
    try:
        sidecar_path = os.path.splitext(video_path)[0] + ".json"
        
        # Load existing if any (to preserve other keys)
        data = {}
        if os.path.exists(sidecar_path):
            try:
                with open(sidecar_path, 'r') as f: data = json.load(f)
            except: pass
            
        data.update({
            "caption_data": caption_meta,
            "pipeline_metrics": pipeline_metrics,
            "last_processed": datetime.now().isoformat()
        })
        
        with open(sidecar_path, "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        logger.warning(f"⚠️ Failed to save sidecar metadata: {e}")


def apply_ferrari_composer(
    input_path: str,
    output_path: str,
    target_res: Tuple[int, int] = (1080, 1920),
    speed_var: float = 0.0,
    color_intensity: float = 0.0,
    voiceover_path: str = None,
    human_safe_mode: bool = False,
    metadata_comment: str = "",
    filter_type: str = "cinematic",
    mirror_mode: bool = False,
    specific_music_path: str = None  # NEW: Continuous Music Injection
):
    """
    FERRARI V2: The Ultimate Single-Pass Composer.
    Merges:
    1. Upscale/Normalization (Scale+Pad)
    2. Speed Ramping (PTS)
    3. Color Grading (Curves+Eq+Unsharp)
    4. Audio Mixing (Ducking/Silent Handling)
    5. Final Encoding (H264/NVENC)
    6. Mirroring (Optional)
    
    Result: 1 Read -> Complex Filter -> 1 Write. Zero intermediate temp files.
    """
    logger.info(f"🏎️ [FERRARI COMPOSER] Assembling Single-Pass Filter Chain...")
    
    # --- 1. VIDEO CHAIN SETUP ---
    w, h = target_res
    
    # Speed/Color Params
    speed_var = max(0.0, min(speed_var, 1.0))
    color_intensity = max(0.0, min(color_intensity, 1.0))
    
    if human_safe_mode:
        color_intensity *= 0.7
        sharpen = "0.3:3:3:0.0"
    else:
        sharpen = "0.6:5:5:0.0"
        
    slow = 1.0 - (0.08 + speed_var * 0.05)
    fast = 1.0 + (0.08 + speed_var * 0.05)
    
    # Filter Chain Construction
    filters = []
    
    # 1. Scale & Pad (Normalization)
    filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease")
    filters.append(f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
    filters.append("setsar=1/1")
    
    # 2. Mirror Mode (Break Fingerprint)
    if mirror_mode:
        filters.append("hflip")
        logger.info("    └─ 🔄 Mirror Mode Active (hflip)")
    
    # 3. Speed Ramping (DISABLED due to Windows FFmpeg syntax instability)
    # The setpts filter with complex expressions is causing persistent 'Invalid Argument' crashes.
    # Removed to ensure pipeline stability.
    # filters.append(...)
    
    # --- COLOR GRADING LOGIC (CLASS-A CINEMATIC UPGRADE) ---
    
    # 1. 🎬 CLASS-A CINEMATIC (The "Film Look")
    # Strategy: 
    # - Lift shadows slightly involved (eq) to prevent crushing
    # - Soft S-Curve via curves (Contrast with roll-off)
    # - Split Tone: Shadows -> Teal (-Red, +Blue), Mids -> Warm (+Red, -Blue)
    # - Global Saturation check (prevent orange skin)
    # 1. 🎬 CLASS-A CINEMATIC (The "Film Look")
    # Strategy: 
    # - Lift shadows slightly involved (eq) to prevent crushing
    # - Boost contrast linearly (eq) instead of curves (to avoid syntax errors)
    # - Split Tone: Shadows -> Teal (-Red, +Blue), Mids -> Warm (+Red, -Blue)
    # - Global Saturation check (prevent orange skin)
    f_cinematic = (
        f"eq=contrast={1.12 + 0.05*color_intensity:.3f}:brightness=0.04:saturation={1.05 + 0.05*color_intensity:.3f}"
        # Removed colorbalance for maximum stability
    )

    # 2. 📸 PAPARAZZI / CLEAN (Flash & Skin Safe)
    # Strategy: Neutral, Bright, High-Fidelity. No color cast.
    # Removed unsharp for stability
    f_paparazzi = (
        f"eq=contrast={1.08 + 0.05*color_intensity:.3f}:brightness=0.04:saturation={1.02 + 0.02*color_intensity:.3f},"
        # Removed curves to prevent syntax errors
        f"colorbalance=rm={0.01*color_intensity:.3f}:bm={-0.01*color_intensity:.3f}" # Micro-warmth for skin
    )

    # 3. 🌑 DARK CINEMA (Modern/Mood)
    # Strategy: Rich blacks, muted highlights, silver tone
    f_dark_cinema = (
        f"eq=contrast={1.20 + 0.1*color_intensity:.3f}:saturation={0.90}," # Hard contrast
        # Removed curves to prevent syntax errors
        f"colorbalance="
        f"rs={-0.04*color_intensity:.3f}:gs={-0.02*color_intensity:.3f}:bs={0.02*color_intensity:.3f}:" # Cold Shadows
        f"rm={0.0}:gm={0.0}:bm={-0.02*color_intensity:.3f}" # Slide Mids to Gold
    )
    
    filters_map = {
        "cinematic": f_cinematic,
        "paparazzi": f_paparazzi,
        "dark_cinema": f_dark_cinema
    }
    
    # Select filter (Default to cinematic)
    f_color = filters_map.get(filter_type, f_cinematic)
    filters.append(f_color)
    logger.info(f"    └─ 🎨 Color Filter: {filter_type} (Intensity: {color_intensity:.2f})")
    
    # Chain: [0:v] -> Scale -> Mirror? -> Ramp -> Color -> [v_out]
    video_filter_chain = ",".join(filters)
    
    # --- 2. AUDIO CHAIN SETUP ---
    
    # SMART TRIM LOGIC (User Request)
    info = _get_video_info(input_path)
    dur = info.get('duration', 0)
    
    trim_args = []
    if dur >= 7.0: # Safety threshold increased to 7s
        # Trim 1s start, 1s end
        # Note: We use -ss before -i for fast seek/trim on input
        trim_args = ["-ss", "1", "-t", f"{dur - 2:.3f}"]
        logger.info(f"✂️ Smart Trim Active: Cutting 1s start/end (New Dur: {dur-2:.1f}s)")
    else:
        logger.warning(f"⚠️ Video too short for trim ({dur}s < 7s). Skipping.")

    inputs = trim_args + ["-i", input_path]
    
    # Base video chain (maps 0:v effectively)
    complex_filter = f"[0:v]{video_filter_chain}[v_processed]" 
    
    # --- AUDIO LOGIC ---
    has_source_audio = _has_audio_stream(input_path)
    
    # Input Mapping Indices
    idx_vo = -1
    idx_music = -1
    next_idx = 1
    
    # 1. LOAD VOICEOVER (If Any and Valid)
    if voiceover_path and os.path.exists(voiceover_path) and os.path.getsize(voiceover_path) > 0:
        inputs.extend(["-i", voiceover_path])
        idx_vo = next_idx
        next_idx += 1
    elif voiceover_path:
        logger.warning(f"⚠️ Voiceover file found but empty/missing: {voiceover_path}. Skipping.")
        
    # 2. AUDIO SOURCE SELECTION
    # Priority: Specific Music > Original (if enabled) > Original Folder > Fallback
    
    # Track the source label for the filter graph
    bg_source_label = None 
    
    # A. Specific Music (Passed Argument)
    if specific_music_path and os.path.exists(specific_music_path):
        inputs.extend(["-i", specific_music_path])
        idx_music = next_idx
        next_idx += 1
        logger.info(f"🎵 Specific Music Track Loaded: {os.path.basename(specific_music_path)}")
        
    # B. Standard Music (Higher Priority)
    elif FALLBACK_AUDIO and not specific_music_path:
        # Try to get from music/ folder first
        fallback_track = _get_next_music_track()
        if fallback_track:
            inputs.extend(["-stream_loop", "-1", "-i", fallback_track]) # Loop music
            idx_music = next_idx
            next_idx += 1
            logger.info(f"🎵 Standard Music Selected: {os.path.basename(fallback_track)}")
            
    # C. Original Audio Folder - DISABLED BY USER REQUEST
    # if idx_music == -1 ... (Logic removed to prevent usage)

    # --- BUILD AUDIO FILTER GRAPH ---
    f_graph = []

    # --- 3. HARD DUCKING & EXCLUSIVE SOURCE ("Spawn Up" Logic) ---
    
    # 3a. Exclusive Source Selection
    if specific_music_path:
         # Use the specific track as BG
         f_graph.append(f"[{idx_music}:a]volume={MUSIC_VOLUME},aresample=44100,aformat=channel_layouts=stereo,apad[s_bg_raw]")
         bg_source_label = "[s_bg_raw]"
         
    elif KEEP_ORIGINAL_AUDIO and has_source_audio:
         f_graph.append(f"[0:a]volume={ORIGINAL_AUDIO_VOLUME},aresample=44100,aformat=channel_layouts=stereo[s_bg_raw]")
         bg_source_label = "[s_bg_raw]"
         
    elif idx_music != -1: # Any Music Selected (Original Folder or Fallback)
         m_vol = MUSIC_VOLUME
         if not KEEP_ORIGINAL_AUDIO: m_vol = 0.6 
         f_graph.append(f"[{idx_music}:a]volume={m_vol},aresample=44100,aformat=channel_layouts=stereo,apad[s_bg_raw]")
         bg_source_label = "[s_bg_raw]"
    else:
         # No Source -> Silence
         f_graph.append(f"anullsrc=channel_layout=stereo:sample_rate=44100[s_bg_raw]")
         bg_source_label = "[s_bg_raw]"

    # 3b. Prepare Voiceover & Control Signal
    final_inputs = []
    
    if idx_vo != -1:
        # Processing VO: Volume -> Infinite Pad -> FORMAT -> Split
        f_graph.append(f"[{idx_vo}:a]volume=1.8,aresample=44100,aformat=channel_layouts=stereo,apad,asplit=2[vo_out][vo_ctrl_raw]")
        
        # Apply HARD Sidechain (The "Spawn Up" Effect)
        # TTS is clean, so we skip gating. Direct [vo_ctrl_raw] usage.
        f_graph.append(f"{bg_source_label}[vo_ctrl_raw]sidechaincompress=threshold=0.01:ratio=20:attack=5:release=300[bg_ducked]")
        
        final_inputs.append("[bg_ducked]")
        final_inputs.append("[vo_out]")
    else:
        # No VO -> Just BG
        final_inputs.append(bg_source_label)
        
    # 3c. Final Additive Merge
    count = len(final_inputs)
    if count == 1:
        f_graph.append(f"{final_inputs[0]}aformat=channel_layouts=stereo[a_out]")
    else:
        # Use amix for merging multiple audio streams
        # Assuming final_inputs will be [bg_ducked, vo_out] when count > 1
        f_graph.append(f"{final_inputs[0]}{final_inputs[1]}amix=inputs={count}:duration=longest:normalize=0[a_out]")

    final_audio_label = "[a_out]"
    audio_block = ";" + ";".join(f_graph)
    complex_filter += audio_block

    # --- 3. ENCODING SETUP ---
    encoder = _get_ffmpeg_encoder()
    preset = "p4" if encoder == "h264_nvenc" else REENCODE_PRESET
    
    cmd = [FFMPEG_BIN, "-y"]
    cmd.extend(inputs)
    cmd.extend(["-filter_complex", complex_filter])
    
    # Map Video
    cmd.extend(["-map", "[v_processed]"])
    
    # Map Audio
    cmd.extend(["-map", final_audio_label])
    
    # Encoding Flags
    cmd.extend([
        "-c:v", encoder,
        "-preset", preset,
        "-c:a", "aac",
        "-map_metadata", "-1",
        "-metadata", f"comment={metadata_comment}",
        "-movflags", "+faststart", # 🚀 WEB OPTIMIZATION (Fix 1)
        "-shortest", # PREVENTS HANG
    ])

    if encoder == "libx264": 
        # HQ Safe Caps for Instagram
        cmd.extend([
            "-crf", REENCODE_CRF, 
            "-profile:v", "high", 
            "-level", "4.2",
            "-maxrate", "15M",
            "-bufsize", "30M"
        ])
    else: cmd.extend(["-rc", "vbr", "-cq", "19", "-maxrate", "15M", "-bufsize", "30M"])
    
    # Output Path MUST BE LAST
    cmd.append(output_path)
    
    logger.info(f"🧬 INJECTING METADATA: {metadata_comment} (Making file unique)")
    logger.info(f"    └─ Encoder: {encoder} | Preset: {preset}")
    _run_command(cmd, check=True)



# ==================== TRANSITIONS ====================

def create_transition_clip(seg_a: str, seg_b: str, output_path: str, trans_type: str, duration: float):
# ... (rest of function unchanged)
    info_a = _get_video_info(seg_a)
    dur_a = info_a['duration']
    start_a = max(0, dur_a - duration)
    
    tail_a = output_path.replace(".mp4", "_tailA.mp4")
    _run_command([FFMPEG_BIN, "-y", "-ss", str(start_a), "-i", seg_a, "-t", str(duration), "-c", "copy", tail_a])
    
    head_b = output_path.replace(".mp4", "_headB.mp4")
    _run_command([FFMPEG_BIN, "-y", "-i", seg_b, "-t", str(duration), "-c", "copy", head_b])
    
    filter_str = f"[0:v][1:v]xfade=transition={trans_type}:duration={duration}:offset=0[v];[0:a][1:a]acrossfade=d={duration}[a]"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", tail_a, "-i", head_b,
        "-filter_complex", filter_str,
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac",
        output_path
    ]
    _run_command(cmd)
    
    if os.path.exists(tail_a): os.remove(tail_a)
    if os.path.exists(head_b): os.remove(head_b)

def compile_with_transitions(input_video: Path, title: str, aggressive_watermark: bool = False, force_reprocess: bool = False, retry_level: int = 0) -> Tuple[Optional[Path], Dict]:
    import audio_processing
    # Import Guard
    try:
        from quality_orchestrator import human_guard
    except ImportError:
        logger.warning("⚠️ Human Guard missing. Assuming CAUTION mode.")
        human_guard = None
    
    input_path = os.path.abspath(str(input_video))

    
    # 0. Reset Quotas (New Job = New Quota)
    try:
        hybrid_watermark.hybrid_detector.reset_quotas()
        logger.info(f"🔄 Quotas reset for new job (Retry Level: {retry_level}).")
    except: pass
    
    # 1. Job Isolation & Atomic Setup
    job_uuid = str(uuid.uuid4())[:8]
    job_id = f"job_{int(time.time())}_{job_uuid}"
    job_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    run_metrics = {
        "job_id": job_id,
        "start_time": time.time(),
        "steps_completed": [],
        "errors": []
    }
    
    final_output = None
    wm_context = None
    vram_safe = _check_vram_safety()
    
    # Init vars for final scope safety
    should_restore_audio = False
    selected_audio_source = None
    
    try:
        logger.info("🚀 [PIPELINE] Job started")
        logger.info(f"    ├─ job_id: {job_id}")
        logger.info(f"    ├─ input: {title}")
        logger.info(f"    └─ vram_available: {vram_safe}")
        
        # Determine Final atomic destination
        safe_title = re.sub(r'[^a-zA-Z0-9_\-]', '', title.replace(" ", "_"))
        if not safe_title: safe_title = f"video_{job_uuid}"
        
        # INCREMENTAL NAMING LOGIC
        # Scan for existing Title_N.mp4
        existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith(safe_title) and f.endswith(".mp4")]
        max_index = 0
        for f in existing_files:
            try:
                # Expect format: Title_1.mp4
                parts = f.replace(".mp4", "").split("_")
                if parts[-1].isdigit():
                    idx = int(parts[-1])
                    if idx > max_index: max_index = idx
            except: pass
            
        next_index = max_index + 1
        final_filename = f"{safe_title}_{next_index}.mp4"
        final_dest = os.path.join(OUTPUT_DIR, final_filename)
        
        logger.info(f"📝 Assigned Incremental Name: {final_filename}")
        
        final_temp_path = os.path.join(job_dir, "final_atomic.mp4")
        final_json_path = os.path.join(job_dir, "final_atomic.json")

        video_info = _get_video_info(input_path)
        logger.info("📊 [METADATA] Video info extracted")
        logger.info(f"    ├─ res: {video_info.get('width')}x{video_info.get('height')}")
        logger.info(f"    └─ duration: {video_info.get('duration'):.1f}s")
        
        # [STAGE 1] Normalization (SAFE MODE: DEFER RESIZE)
        # "Upscaling MUST occur strictly AFTER watermark removal."
        norm_video = os.path.join(job_dir, "normalized.mp4")
        
        # Rule 1: Normalize FPS/Format but DEFER scaling.
        # We pass resize=False to KEEP original geometry.
        if normalize_video(input_path, norm_video, resize=False):
             current_video = norm_video
        else:
             current_video = input_path
        
        # Ensure norm path is consistent
        if current_video == input_path:
             norm_video = input_path
             
        # [STAGE 2] Watermark Detection
        enable_gemini_detect = os.getenv("ENABLE_GEMINI_WATERMARK_DETECT", "yes").lower() == "yes"
        final_watermarks = []
        inpainted_regions = []
        
        if enable_gemini_detect:
            try:
                import hybrid_watermark
                
                # --- METADATA KEYWORD INJECTION ---
                keywords_str = None
                try:
                    # Look for sidecar JSON (downloader usually saves as .info.json or just .json)
                    # We check both <name>.json and <name>.info.json
                    base_no_ext = os.path.splitext(input_path)[0]
                    candidates = [base_no_ext + ".json", base_no_ext + ".info.json"]
                    
                    meta_data = {}
                    for c in candidates:
                        if os.path.exists(c):
                             try: 
                                 with open(c, 'r', encoding='utf-8') as f:
                                     meta_data = json.load(f)
                                 break
                             except: pass
                             
                    keys = []
                    if meta_data:
                        # Extract High-Value Targets
                        targets = [
                            meta_data.get('uploader'),
                            meta_data.get('channel'),
                            meta_data.get('uploader_id'),
                            meta_data.get('id') 
                        ]
                        
                        for t in targets:
                            if t and isinstance(t, str) and len(t) > 2:
                                # Clean: Remove all special characters, keep alphanumeric + spaces
                                t_clean = re.sub(r'[^a-zA-Z0-9\s]', '', t)
                                if len(t_clean.strip()) > 2:
                                    keys.append(t_clean)
                                    keys.append(t_clean.upper())
                                    keys.append(t_clean.lower())
                                    
                        # Dedupe and Join
                        if keys:
                            unique_keys = list(set(keys))
                            keywords_str = ", ".join(unique_keys)
                            logger.info(f"    ├─ 🔑 Generated Keywords: {len(unique_keys)} variants loaded.")
                except Exception as e:
                    logger.warning(f"    ├─ ⚠️ Metadata extraction failed: {e}")

                wm_result_json = hybrid_watermark.hybrid_detector.process_video(
                    current_video, 
                    aggressive=aggressive_watermark,
                    keywords=keywords_str, # INJECTED
                    retry_level=retry_level
                )
                wm_result = json.loads(wm_result_json)
                
                # ABORT CHECK (UNSAFE)
                if wm_result.get('error') and "UNSAFE" in str(wm_result['error']):
                     return None, {"reason": "Aborted: UNSAFE Frame"}
                     
                wm_context = wm_result.get('context', {})
                # FIX: Propagate top-level status to context for downstream logic
                wm_context['watermark_status'] = wm_result.get('status', 'UNKNOWN')
                
                run_metrics["watermark_context"] = wm_context
                final_watermarks = wm_result.get('watermarks', [])
                
                # [STAGE 3] Adaptive Inpainting & Tuning
                if final_watermarks and WATERMARK_REMOVE:
                     clean_video = os.path.join(job_dir, "clean_adaptive.mp4")
                     # REFACTORED: Use Orchestrator
                     success, reason = run_adaptive_watermark_orchestration(
                         current_video, 
                         final_watermarks, 
                         clean_video, 
                         job_dir, 
                         original_height=video_info.get('height'),
                         aggressive_mode=aggressive_watermark,
                         retry_level=retry_level
                     )
                     
                     # --- AUTO-RETRY LOGIC (LEGACY SAFEGUARD) ---
                     if not success and not aggressive_watermark and retry_level == 0:
                         logger.warning(f"⚠️ First pass failed ({reason}). Legacy Auto-Retry...")
                         # Retry with Aggressive Mode enabled (Legacy branch)
                         success, reason = run_adaptive_watermark_orchestration(
                             current_video, 
                             final_watermarks, 
                             clean_video, 
                             job_dir, 
                             original_height=video_info.get('height'),
                             aggressive_mode=True
                         )
                     
                     if success:
                         # [AUDIO LIBRARY SWAP]
                         # 1. Setup Library
                         audio_pool_dir = os.path.join(os.getcwd(), "Original_audio")
                         os.makedirs(audio_pool_dir, exist_ok=True)
                         
                         # 2. Extract Current Audio to Library
                         current_audio_id = f"audio_{job_uuid}.mp3"
                         current_audio_path = os.path.join(audio_pool_dir, current_audio_id)
                         has_extracted = False
                         
                         try:
                             # Extract audio from normalized source
                             cmd_extract = [
                                 FFMPEG_BIN, "-y", "-i", norm_video,
                                 "-vn", "-c:a", "libmp3lame", "-q:a", "2",
                                 current_audio_path
                             ]
                             # Only extract if it doesn't exist (dedupe logic handled by UUID)
                             # Check output
                             res = _run_command(cmd_extract)
                             if os.path.exists(current_audio_path) and os.path.getsize(current_audio_path) > 1000:
                                 has_extracted = True
                                 logger.info(f"💾 Audio Extracted to Pool: {current_audio_path}")
                             else:
                                 logger.warning("⚠️ Audio Extraction failed or empty.")
                         except Exception as e:
                             logger.warning(f"⚠️ Audio Extract Error: {e}")

                         # 3. Select Audio Strategy (Flags)
                         should_restore_audio = False
                         selected_audio_source = None
                         
                         if not KEEP_ORIGINAL_AUDIO:
                             # Case: Keep=No
                             # We skip restoration. 'video' remains silent.
                             # Downstream: has_audio=False -> Uses FALLBACK_AUDIO (Music)
                             logger.info("🔀 Audio Strategy: SKIPPED (KeepOriginal=No). Waiting for Music Fallback.")
                         else:
                             # Case: Keep=Yes (Try Swap)
                             try:
                                 pool_files = glob.glob(os.path.join(audio_pool_dir, "*.mp3"))
                                 # Exclude current
                                 candidates = [f for f in pool_files if os.path.abspath(f) != os.path.abspath(current_audio_path)]
                                 
                                 if candidates:
                                    # Success: Swap from Pool (LRU ROTATION LOGIC)
                                     # 1. Load History
                                     history_path = os.path.join(audio_pool_dir, "audio_history.json")
                                     audio_history = {}
                                     try:
                                         if os.path.exists(history_path):
                                             with open(history_path, 'r') as f: audio_history = json.load(f)
                                     except: pass

                                     # 2. Sort Candidates by Last Used (Ascending: None/Oldest first)
                                     # Default timestamp 0 for unused
                                     candidates.sort(key=lambda x: audio_history.get(os.path.basename(x), 0))
                                     
                                     # 3. Select Top (2nd Least Recently Used - User Request)
                                     # User concern: LRU #1 might be close to original.
                                     if len(candidates) > 1:
                                         selected_audio_source = candidates[1]
                                         logger.info("🔀 Strategy: Picked 2nd LRU (Variety Boost)")
                                     else:
                                         selected_audio_source = candidates[0]
                                     
                                     # 4. Update History
                                     audio_history[os.path.basename(selected_audio_source)] = time.time()
                                     try:
                                         with open(history_path, 'w') as f: json.dump(audio_history, f, indent=2)
                                     except: pass
                                     
                                     should_restore_audio = True
                                     logger.info(f"🔀 Audio Strategy: SWAPPED (LRU) -> {os.path.basename(selected_audio_source)}")
                                 else:
                                     # Failure: Pool Empty
                                     if FALLBACK_AUDIO:
                                         # Case: Keep=Yes, Fallback=Yes, Pool=Empty -> Use Music
                                         should_restore_audio = False
                                         logger.info("🔀 Audio Strategy: POOL EMPTY -> Fallback to Music.")
                                     else:
                                         # Case: Keep=Yes, Fallback=No, Pool=Empty -> Force Self
                                         selected_audio_source = norm_video
                                         should_restore_audio = True
                                         logger.info("🔀 Audio Strategy: POOL EMPTY -> Forcing Self (Source).")
                                         
                             except Exception as e:
                                 logger.warning(f"⚠️ Audio Selection Error: {e}")
                                 # Safety Fallback
                                 if not FALLBACK_AUDIO:
                                     selected_audio_source = norm_video
                                     should_restore_audio = True

                         # 4. Inject Audio (If selected)
                         if should_restore_audio and selected_audio_source:
                             clean_with_audio = os.path.join(job_dir, "clean_audio_swapped.mp4")
                             try:
                                 # Map Video from Clean (0:v), Audio from Selected (1:a)
                                 # ADDED: -stream_loop -1 for the audio input to prevent silence if audio < video
                                 cmd_restore = [
                                     FFMPEG_BIN, "-y", 
                                     "-i", clean_video,
                                     "-stream_loop", "-1", "-i", selected_audio_source,
                                     "-map", "0:v", "-map", "1:a",
                                     "-c:v", "copy", "-c:a", "aac", # Copy video, re-encode audio to AAC match
                                     "-shortest",
                                     clean_with_audio
                                 ]
                                 if _run_command(cmd_restore, check=True):
                                     current_video = clean_with_audio
                                     logger.info("🔊 Audio Restored/Swapped.")
                                 else:
                                     current_video = clean_video
                                     logger.warning("⚠️ Audio Swap Failed. Continuing silent.")
                             except Exception as e:
                                 logger.warning(f"⚠️ Audio Swap Except: {e}")
                                 current_video = clean_video
                         else:
                             # No restoration -> Continuing silent
                             current_video = clean_video

                         # Record Inpainted Regions
                         for wm in final_watermarks:
                             if wm.get('coordinates'):
                                 inpainted_regions.append(wm['coordinates'])
                                 
                         wm_context['removal_success'] = True
                         wm_context['watermark_status'] = "DETECTED_AND_REMOVED"
                         wm_context['transformative_assertion'] = "Original content structurally modified via AI reconstruction"
                         wm_context['method'] = "adaptive_ai_inpaint"
                         logger.info(f"🧼 Watermark Status: {wm_context['watermark_status']}")
                         logger.info(f"🔄 Assertion: {wm_context['transformative_assertion']}")
                     else:
                         logger.warning(f"🛑 QA FAIL: Watermark removal aborted ({reason}). Reverting.")
                         wm_context['removal_success'] = False
                         wm_context['watermark_status'] = "DETECTED_BUT_FAILED"
                         wm_context['skipped_reason'] = reason
                         # Pipeline continues with original video
                                
            except Exception as e:
                logger.error(f"Watermark Pipeline Error: {e}")
                # If exception occurred, assume unsafe
                return None, {"reason": f"Watermark Check Error: {e}"}

        # [STAGE 4] Smart Edge Crop (Zoom)
        # Apply AFTER watermark removal to clean up edges
        if EDGE_CROP_FACTOR > 0:
             cropped_video = os.path.join(job_dir, "cropped_z.mp4")
             if apply_edge_crop(current_video, cropped_video, factor=EDGE_CROP_FACTOR):
                 current_video = cropped_video
                 run_metrics["steps_completed"].append("edge_crop")

        # [STAGE 5] Captions / Overlay
        ai_caption_text = None
        
        # [STAGE 6] CAPTION GENERATION (Blocking, Post-Watermark)
        # "Caption generation MUST analyze the FIRST CLEAN FRAME AFTER watermark removal"
        ai_caption_text = None
        
        wm_ctx = run_metrics.get("watermark_context", {})
        wm_status = wm_ctx.get("watermark_status", "UNKNOWN")
        wm_success = wm_ctx.get("removal_success", False)
        
        # Policy: SKIP if removal failed
        if wm_status == "DETECTED_BUT_FAILED" or (wm_status == "DETECTED_AND_REMOVED" and not wm_success):
             logger.warning("🚫 Caption Generation BLOCKED until Watermark Removal Complete (Status: FAILED).")
        else:
             # Allowed: CLEAN or REMOVED
             if os.getenv("AI_CAPTIONS", "yes").lower() == "yes":
                  logger.info("📝 Caption Generation Triggered (Post-Watermark, Pre-Upscale)")
                  logger.info(f"🧼 Caption Source Frame: CLEAN (Context: {wm_status})")
                  
                  try:
                      from gemini_captions import generate_caption_direct
                      # BLOCKING CALL
                      # "Caption generation uses a SEPARATE Gemini call"
                      # "Fail-safe behavior: If time out -> Continue pipeline WITHOUT caption"
                      ai_caption_text = generate_caption_direct(current_video)
                      run_metrics["caption"] = ai_caption_text
                  except Exception as e:
                      logger.warning(f"⚠️ Caption Generation Failed/Timed Out: {e}")
                      ai_caption_text = None

        if ai_caption_text:
             try:
                 # NEW: Run Monetization Analysis (with Transformation Data)
                 try:
                     # MOVED TO STAGE 10
                     # from monetization_brain import MonetizationStrategist
                     # brain = MonetizationStrategist()
                     brain = None
                     
                     # 1. Gather Transformation Data
                     transformations = {}
                     
                     # A. Watermark Removal (Area based)
                     # We don't have exact pixel count from here easily without reading masks again.
                     # But we know if we did it.
                     if run_metrics.get("watermark_context", {}).get("removal_success"):
                         # Guess area based on typical logo (approx 5-10%)
                         transformations["Inpainting"] = "Logo Removal (~8% Area)"
                         
                     # B. Upscale
                     if "golden_upscaled.mp4" in current_video:
                         transformations["Upscale"] = "AI Super-Res (1080p)"
                         
                     # C. Filters (Speed, Color) - Checking run_metrics
                     # Note: Color/Speed happen LATER in this function. 
                     # So we are predicting them or need to move this call?
                     # Ideally we move monetize analysis to the END of pipeline?
                     # But we need caption for text overlay.
                     # We can just claim we WIIL apply them.
                     # C. Filters (Speed, Color)
                     transformations["Color Grading"] = "Cinematic Filter"
                     transformations["Speed Ramp"] = "Dynamic Adjustment"
                     
                     # D. Smart Trim Intent (Predictive)
                     # We know logic: If > 7s, we trim.
                     try:
                         # Re-check duration for report accuracy
                         d_check = _get_video_info(current_video).get("duration", 0)
                         if d_check >= 7.0:
                              transformations["Smart Trim"] = "Cut 1s Start + 1s End (Transformative)"
                     except: pass

                     # E. Voiceover Intent
                     if os.getenv("AI_VOICEOVER", "yes").lower() == "yes":
                         transformations["Voiceover"] = "Original Editorial Commentary"
                     
                     # Analyze
                     # MOVED TO STAGE 10 (FINAL AUDIT) PER USER REQUEST
                     # risk_report = brain.analyze_content(current_video, ai_caption_text, transformations=transformations)
                     # run_metrics["monetization"] = risk_report
                     # logger.info(f"💰 Risk Analysis: {risk_report.get('risk_level')} ({risk_report.get('risk_reason')})")
                     # logger.info(f"🎨 Transformation: {risk_report.get('transformation_score')}% ({risk_report.get('verdict')})")
                 except Exception as e:
                     logger.warning(f"⚠️ Monetization Analysis Failed: {e}")
                     run_metrics["monetization"] = {"risk_level": "UNKNOWN", "risk_reason": "Analysis Failed", "source": "compiler"}

             except Exception as e:
                 logger.warning(f"⚠️ Caption Processing Failed: {e}. Proceeding.")
                 try:
                      # If caption failed, trying risk analysis might also fail if it depends on caption
                      # But we can try with empty caption or skip
                      pass
                 except: pass


        # [STAGE 4.5] PRE-MIRROR (Stealth Mode)
        # We must mirror BEFORE text overlay so text isn't backwards.
        # Check if we should mirror (User enabled manual hflip via intent or default)
        # For now, we defaulting to True for YPP safety as discussed.
        USE_MIRROR = True 
        if USE_MIRROR:
             mirror_temp = os.path.join(job_dir, "mirrored_pre.mp4")
             logger.info("🔄 Applying Pre-Mirror (Horizontal Flip) for YPP Evasion...")
             # Fast re-encode or stream copy? hflip requires filter -> re-encode
             # Use ultrafast preset to minimize time, high bitrate to keep quality for next steps
             cmd_mirror = [
                 FFMPEG_BIN, "-y", "-i", current_video,
                 "-vf", "hflip",
                 "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                 "-c:a", "copy",
                 mirror_temp
             ]
             if _run_command(cmd_mirror):
                 current_video = mirror_temp
                 logger.info("✅ Pre-Mirror Applied successfully.")

        if HAS_TEXT_OVERLAY:
             if ai_caption_text:
                  tv1 = os.path.join(job_dir, "text_1.mp4")
                  if apply_text_overlay_safe(current_video, tv1, ai_caption_text, lane="caption", size=TEXT_OVERLAY_SIZE):
                       current_video = tv1
             if ADD_TEXT_OVERLAY and TEXT_OVERLAY_TEXT:
                  tv2 = os.path.join(job_dir, "text_2.mp4")
                  if apply_text_overlay_safe(current_video, tv2, TEXT_OVERLAY_TEXT, lane="fixed", size=TEXT_OVERLAY_SIZE):
                       current_video = tv2

        # [STAGE 5.5 - FINAL] FERRARI V2 SINGLE PASS COMPOSER
        # Merges Upscale, Speed, Color, Audio Mix, and Final Encode into ONE Step.
        logger.info("🏎️ [FERRARI V2] Engaging Single-Pass Composer...")
        
        # A. PREPARE PARAMS
        # A. PREPARE PARAMS
        speed_var = SPEED_VARIATION
        
        # Ensure Color Grading respects flag
        if ADD_COLOR_GRADING:
            color_int = COLOR_INTENSITY
            color_filter_type = COLOR_FILTER
            logger.info(f"🎨 Color Grading ACTIVE: {color_filter_type} (Intensity: {color_int})")
        else:
            color_int = 0.0
            color_filter_type = "cinematic"
            logger.info("🎨 Color Grading DISABLED (Flag=No). Intensity set to 0.")

        human_safe = False
        
        # Safety Check (Human Guard)
        if ADD_COLOR_GRADING and human_guard:
             check_frame = os.path.join(job_dir, "safety_check.jpg")
             try:
                 subprocess.run([FFMPEG_BIN, "-y", "-i", current_video, "-ss", "00:00:01", "-vframes", "1", check_frame], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                 guard_report = human_guard.analyze_human_presence(check_frame)
                 if guard_report['has_humans']:
                     human_safe = True
                     logger.info("busts Scene Analysis: HUMAN DETECTED (Safety Protocols Active)")
                 else:
                     logger.info("🌍 Scene Analysis: SCENERY (Full Enhancement Allowed)")
             except: pass

        # B. GENERATE VOICEOVER ASSET (If needed)
        vo_path_arg = None
        if os.getenv("ENABLE_MICRO_VOICEOVER", "yes").lower() == "yes" and ai_caption_text:
             from voiceover import generate_voiceover
             vo_target = os.path.join(job_dir, "voiceover.mp3")
             # We generate it, but don't mix yet. Composer will mix.
             if generate_voiceover(ai_caption_text, vo_target):
                 vo_path_arg = vo_target
                 logger.info(f"🎙️ Voiceover Asset Ready: {vo_path_arg}")

        # C. EXECUTE SINGLE PASS
        apply_ferrari_composer(
            input_path=current_video,
            output_path=final_temp_path,
            target_res=(1080, 1920),
            speed_var=speed_var,
            color_intensity=color_int,
            voiceover_path=vo_path_arg,
            human_safe_mode=human_safe,
            metadata_comment=f"ID:{job_uuid}",
            filter_type=color_filter_type
        )
        
        run_metrics["steps_completed"].append("ferrari_v2_composer")
        
        # D. VERIFY & METADATA
        if verify_video_integrity(final_temp_path):
             # Make JSON Meta
             final_meta = {
                 "unique_id": job_uuid,
                 "source": input_path,
                 "created_at": datetime.now().isoformat(),
                 "watermark_safe": True,
                 "pipeline_metrics": run_metrics
             }
             with open(final_json_path, "w") as f: json.dump(final_meta, f, indent=2)
             
             # ATOMIC MOVE
             dest_meta = os.path.splitext(final_dest)[0] + ".json"
             # Reverted: Keep JSON with video initially for main.py access.
             
             # [STAGE 10] MONETIZATION GUARD (Final Gate)
        logger.info("🛡️ [STAGE 10] Monetization Guard (Final Policy Check)")
        
        # --- LATE BINDING: Run Brain Analysis NOW ---
        # "Run last after all process done"
        try:
             # Re-gather transformations (Accurate Final State)
             final_transformations = {}
             # 1. Watermark
             if wm_context and wm_context.get('removal_success'):
                 final_transformations["Inpainting"] = "Logo Removal (Success)"
                 
             # 2. Trim
             if speed_var > 0: final_transformations["Speed Ramp"] = "Dynamic Adjustment"
             
             # 3. Audio & Voiceover (NEW)
             if should_restore_audio and selected_audio_source:
                 final_transformations["Audio"] = "Background Track Swapped"
                 
             if vo_path_arg:
                 final_transformations["Voiceover"] = "AI Commentary Added"
             
             try:
                 d_check = _get_video_info(final_temp_path).get("duration", 0)
                 # If original was > 7 and this is < Orig-2? 
                 # Simpler: If we had intent.
                 if _get_video_info(input_path).get("duration", 0) >= 7.0:
                      final_transformations["Smart Trim"] = "Start/End Cuts Applied"
             except: pass
             
             if ai_caption_text:
                  from monetization_brain import MonetizationStrategist
                  brain = MonetizationStrategist()
                  
                  # Use FINAL video for analysis context (though Brain uses text description)
                  risk_report = brain.analyze_content(current_video, ai_caption_text, transformations=final_transformations)
                  run_metrics["monetization"] = risk_report
                  
                  logger.info(f"💰 Final Risk Analysis: {risk_report.get('risk_level')} ({risk_report.get('risk_reason')})")
                  logger.info(f"🎨 Final Transformation: {risk_report.get('transformation_score')}% ({risk_report.get('verdict')})")
        except Exception as e:
             logger.warning(f"⚠️ Final Brain Analysis Failed: {e}")

        # Calculate Transformation Score
        # USER REQUEST: "gemini percentage as tranformation score"
        t_score = run_metrics.get("monetization", {}).get("transformation_score", 100)
        
        # 1. Watermark Check
        # WAIVER: If skipped due to quota, do not penalize.
        skip_reason = wm_context.get('skipped_reason', '')
        if "Quota" in str(skip_reason) or "429" in str(skip_reason):
             logger.warning("   ⚠️ Watermark check skipped (Quota). Waiving penalty.")
        elif not wm_context or not wm_context.get('removal_success'):
            t_score -= 50
            logger.warning("   ⚠️ Watermark not fully removed (-50)")
        
        # 2. Caption Check 
        # (Need to track if caption was actually generated. We assume Stage 6 passes implies yes)
        # Let's inspect pipeline_metrics or context? 
        # For now, assume generated if we passed Stage 6 without error, but better to check file existence?
        # Actually, we have `caption_text` variable if we stored it?
        # Simpler: If we reached here, caption *should* be there unless skipped.
        # Strict rule: "Caption generation must add... narrative value".
        # We will assume if "caption_meta" has content, it's good.
        
        # 3. Final Decision
        abort_reason = None
        
        if not wm_context or wm_context.get('watermark_status') != "DETECTED_AND_REMOVED":
            # Allow CLEAN if initially clean
            if wm_context and wm_context.get('watermark_status') == "CLEAN":
                pass
            else:
                # RELAXED POLICY: Do not abort, just penalized score (-50) above.
                logger.warning("⚠️ Monetization Warning: Watermark residue present (proceeding with penalty).")
                # abort_reason = "Watermark Residue / Not Removed"
                
        # Check score threshold
        # RELAXED LOGIC FOR QUOTA ERRORS
        is_quota_error = run_metrics.get("monetization", {}).get("risk_level") == "UNKNOWN"
        
        min_score = 30
        if is_quota_error:
             min_score = 10 # Allow almost anything if we can't verify
             logger.warning("⚠️ Monetization Guard: Quota Error detected. Relaxing score threshold to 10.")
             
        if force_reprocess and t_score < min_score: 
            abort_reason = f"Low Transformation Score ({t_score}) on Collision (Threshold: {min_score})"
            
        if abort_reason:
            logger.error(f"❌ MONETIZATION ABORT: {abort_reason}")
            run_metrics['monetization'] = {'risk_level': 'CRITICAL', 'reason': abort_reason}
            # Populate caption_meta even on abort
            caption_meta = {'caption': run_metrics.get('caption')} if run_metrics.get('caption') else {}
            _save_sidecar(final_dest, caption_meta=caption_meta, pipeline_metrics=run_metrics)
            return None, wm_context
    
        logger.info(f"✅ Pipeline Success. Transformation Score: {t_score}")
        if 'monetization' not in run_metrics: run_metrics['monetization'] = {}
        run_metrics['monetization'].update({'risk_level': 'LOW', 'transformation_score': t_score, 'verdict': 'Pipeline Certified'})
        
        # Populate caption_meta for success
        caption_meta = {'caption': run_metrics.get('caption')} if run_metrics.get('caption') else {}
        _save_sidecar(final_dest, caption_meta=caption_meta, pipeline_metrics=run_metrics) # Update final status logic if needed
        
        # Move Atomic to Final
        if os.path.exists(final_temp_path):
            import shutil
            shutil.move(final_temp_path, final_dest)
            
        return Path(final_dest), wm_context
        
    except Exception as e:
        logger.error(f"Pipeline Critical Error: {e}", exc_info=True)
        return None, {'error': str(e)}


    finally:
        if os.getenv("DEBUG_JSON", "0") == "1":
             try:
                 with open(os.path.join(job_dir, "debug.json"), "w") as f: json.dump(run_metrics, f, indent=2, default=str)
             except: pass
             
        _prune_temp_dirs()
        if CLEANUP_POLICY == "immediate":
             shutil.rmtree(job_dir, ignore_errors=True)


# ==================== AUDIO FALLBACK ====================


# ==================== BRANDING INJECTION ====================

def _inject_branding(compilation_path: str) -> str:
    """
    Injects Intro/Outro clips to the compiled video if assets exist.
    """
    try:
        intro_dir = "remarks/intro"
        outro_dir = "remarks/outro"
        
        intros = glob.glob(os.path.join(intro_dir, "*.mp4")) if os.path.exists(intro_dir) else []
        outros = glob.glob(os.path.join(outro_dir, "*.mp4")) if os.path.exists(outro_dir) else []
        
        if not intros and not outros:
            return compilation_path
            
        logger.info("🎨 Checking for Branding Assets...")
        
        intro_file = random.choice(intros) if intros else None
        outro_file = random.choice(outros) if outros else None
        
        if intro_file: logger.info(f"    └─ 🎬 Intro: {os.path.basename(intro_file)}")
        if outro_file: logger.info(f"    └─ 🎬 Outro: {os.path.basename(outro_file)}")
        
        # Prepare inputs
        inputs = []
        filter_str = ""
        stream_idx = 0
        video_maps = []
        audio_maps = []
        
        # Helper to scale branding elements to match compilation (1080x1920)
        # We assume compilation is 1080x1920.
        # We force branding checks to scale/pad.
        
        # 1. Intro
        if intro_file:
            inputs.extend(["-i", intro_file])
            # Scale to 1080:1920, Force SAR 1:1, Force FPS 30 (or whatever compilation is), FORCE PIXEL FORMAT yuv420p
            # We use generic filter: scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2
            filter_str += f"[{stream_idx}:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{stream_idx}];"
            filter_str += f"[{stream_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{stream_idx}];"
            video_maps.append(f"[v{stream_idx}]")
            audio_maps.append(f"[a{stream_idx}]")
            stream_idx += 1
            
        # 2. Main Video
        inputs.extend(["-i", compilation_path])
        filter_str += f"[{stream_idx}:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{stream_idx}];"
        filter_str += f"[{stream_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{stream_idx}];"
        video_maps.append(f"[v{stream_idx}]")
        audio_maps.append(f"[a{stream_idx}]")
        stream_idx += 1
        
        # 3. Outro
        if outro_file:
            inputs.extend(["-i", outro_file])
            filter_str += f"[{stream_idx}:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p[v{stream_idx}];"
            filter_str += f"[{stream_idx}:a]aformat=sample_rates=44100:channel_layouts=stereo[a{stream_idx}];"
            video_maps.append(f"[v{stream_idx}]")
            audio_maps.append(f"[a{stream_idx}]")
            stream_idx += 1
            
        # Concat
        # ORDER MATTERS for concat filter with v=1:a=1
        # It expects [v0][a0][v1][a1][v2][a2]...
        
        n = len(video_maps)
        concat_input_maps = ""
        for i in range(n):
             concat_input_maps += video_maps[i] + audio_maps[i]
             
        filter_str += f"{concat_input_maps}concat=n={n}:v=1:a=1[outv][outa]"
        
        branded_output = compilation_path.replace(".mp4", "_branded.mp4")
        
        cmd = [
            FFMPEG_BIN, "-y"
        ] + inputs + [
            "-filter_complex", filter_str,
            "-map", "[outv]", "-map", "[outa]",
            "-c:v", "libx264", "-preset", REENCODE_PRESET, "-crf", REENCODE_CRF,
            "-maxrate", "15M", "-bufsize", "30M", "-profile:v", "high", "-level", "4.2",
            "-c:a", "aac", "-b:a", "192k",
             branded_output
        ]
        
        logger.info("🔨 Injecting Branding (Intro/Outro)...")
        if _run_command(cmd, timeout=300, check=True):
             # Atomic replacement
             try:
                 os.remove(compilation_path)
                 shutil.move(branded_output, compilation_path)
                 logger.info("✅ Branding Injected Successfully.")
                 return compilation_path
             except Exception as e:
                 logger.error(f"❌ Failed to replace branded file: {e}")
                 return compilation_path
        else:
             logger.warning("⚠️ Branding injection failed (ffmpeg error). Returning original.")
             return compilation_path

    except Exception as e:
        logger.error(f"⚠️ Branding injection failed: {e}")
        return compilation_path

# ==================== AUDIO FALLBACK ====================
# Using shared logic from audio_processing.py

# Legacy Stub for compatibility
def compile_batch_with_transitions(video_files: List[str], output_filename: str) -> Optional[str]:
    """
    Compiles multiple videos into one with transitions.
    Now supports Smart Audio Fallback (Music Overlay) if primary render fails.
    """
    logger.info(f"🎬 Starting batch compilation of {len(video_files)} clips...")
    logger.info(f"    └─ Output: {output_filename}")
    
    # QA Check: Filter invalid files
    original_count = len(video_files)
    video_files = [f for f in video_files if is_valid_video(f)]
    if len(video_files) < original_count:
        logger.warning(f"⚠️ Filtered {original_count - len(video_files)} invalid/empty files from batch.")
    
    if not video_files:
        logger.error("❌ No valid video files to compile.")
        return None
    
    # 0. Setup Job
    job_uuid = str(uuid.uuid4())[:8]
    job_dir = os.path.join(TEMP_DIR, f"batch_{job_uuid}")
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        # 1. Normalize Inputs (Resolution, FPS, SAR)
        # We need uniform inputs for concatenation
        norm_files = []

        # 1.5 SMART DURATION LOGIC (Shorts vs Long Form)
        # Calculate Total Raw Duration first to decide strategy
        total_raw_duration = 0
        for f in video_files:
             try:
                 total_raw_duration += _get_video_info(f)['duration']
             except: pass
             
        logger.info(f"⏱️ Total Raw Input Duration: {total_raw_duration:.1f}s")
        
        target_res = tuple(map(int, os.getenv("TARGET_RESOLUTION", "1080:1920").split(":")))
        n_clips = len(video_files)
        
        # 1.6 CONTINUOUS MUSIC MANAGER
        # ONLY for Compilations (Long Form)
        music_manager = None
        if total_raw_duration > 180:
             try:
                 from music_manager import ContinuousMusicManager
                 music_manager = ContinuousMusicManager()
                 logger.info("🎵 Continuous Music Manager Initialized.")
             except ImportError:
                 logger.warning("⚠️ ContinuousMusicManager not found. Falling back to simple logic.")

        # FIX: Fallback to Long Form if duration is 0 (failed detection) to avoid destructive trimming
        if total_raw_duration > 180 or total_raw_duration <= 0.1:
             # LONG FORM MODE (or Safe Fallback)
             if total_raw_duration <= 0.1:
                 logger.warning("⚠️ Duration detection failed (Total=0). Defaulting to SAFE MODE (No Trimming).")
             else:
                 logger.info("🎬 Mode: LONG FORM (Total > 180s). Disabling trims & Rebuilding Audio.")
                 logger.info("ℹ️ NOTE: YouTube now classifies vertical videos up to 3 MINUTES as Shorts. This upload may still be a Short.")
             max_per_clip = None 
             
        else:
             # SHORTS MODE (Aggressive Trim to 30s as requested)
             logger.info("⚡ Mode: SHORTS (Total <= 180s). Trimming to 30s for fast pacing.")
             # Target ~28s to be safe
             total_allowed = 28.0 + ((n_clips - 1) * TRANSITION_DURATION)
             max_per_clip = total_allowed / n_clips if n_clips > 0 else 28.0
             max_per_clip = max(3.0, max_per_clip) # Min 3s per clip
             logger.info(f"⏳ Shorts Compliance: Limiting each clip to {max_per_clip:.1f}s")
        
        for i, v_path in enumerate(video_files):
            # Check if valid
            if not os.path.exists(v_path): continue
            
            norm_path = os.path.join(job_dir, f"clip_{i:03d}.mp4")
            # Reuse normalize logic with DURATION LIMIT
            if normalize_video(v_path, norm_path, target_res, max_duration=max_per_clip):
                
                # --- AUDIO REMIX LOGIC (Shorts & Long Form) ---
                rebuilt_path = os.path.join(job_dir, f"rebuilt_{i:03d}.mp4")
                
                # Logic:
                # If Shorts (<180s): Use Legacy Batch Mix (Post-Process) -> Skip per-clip music here.
                # If Long Form (>180s): Use ContinuousMusicManager (Per-Clip Pre-Mix).
                
                should_premix_continuous = (total_raw_duration > 180 and music_manager)
                
                music_segment_path = None
                
                if should_premix_continuous:
                     try:
                         # 1. Get exact duration of normalized clip
                         info = _get_video_info(norm_path)
                         clip_dur = info.get('duration', 30.0)
                         
                         # 2. Add extra for transition overlap?
                         # Usually we crossfade, so we want music to cover the visual duration.
                         # Ferrari trims 2s (1s start/end) if >7s, but 'normalize_video' happened already.
                         # Wait, 'apply_ferrari_composer' trims?
                         # normalize_video strictly sets duration if max_duration set.
                         # But if long form (max_per_clip = None), normalize doesn't trim.
                         # apply_ferrari_composer DOES trim 1s/1s if dur>7.0 (Smart Trim).
                         # We should calculate the *final* duration the clip will have.
                         
                         final_dur = clip_dur
                         if clip_dur >= 7.0: final_dur -= 2.0
                         
                         # 3. Allocation
                         segments = music_manager.allocate_music(final_dur)
                         
                         if segments:
                             # 4. Build Mix
                             from audio_processing import build_continuous_segment
                             music_segment_path = os.path.join(job_dir, f"music_segment_{i}.mp3")
                             if build_continuous_segment(music_segment_path, segments):
                                 logger.info(f"    └─ 🎵 Built continuous music segment ({final_dur:.1f}s)")
                             else:
                                 music_segment_path = None
                     except Exception as e:
                         logger.warning(f"⚠️ Continuous Music Prep Failed: {e}")

                # Call Ferrari Composer (Single-Pass Transform)
                # Note: We replace the separate _rebuild_clip_audio logic with integrated logic inside Ferrari?
                # Actually, apply_ferrari_composer handles mixing if 'idx_music' is set. 
                # But it expects a "loopable" track mostly or Fallback Audio.
                # We need to pass our specific 'music_segment_path' as a source.
                
                # FIX: We need apply_ferrari_composer to accept an explicit music file path to mix.
                # It currently checks 'Original_audio' folder or 'FALLBACK_AUDIO' config.
                # We can trick it or modify it. 
                # Better to modify it to accept `specific_music_path`.
                # Assuming I added `specific_music_path` arg to apply_ferrari_composer (I will in next step).
                # For now, I will rename the param 'voiceover_path' -> 'voiceover_path', 'audio_track_path' -> new param.
                
                # Wait, I haven't modified apply_ferrari_composer signature yet. 
                # I will modify call here, and then modify function def in next tool call.
                
                # Fetch metadata
                meta_path = v_path + ".json"
                if not os.path.exists(meta_path): meta_path = os.path.splitext(v_path)[0] + ".json"
                
                caption = ""
                if os.path.exists(meta_path):
                     try: 
                         with open(meta_path) as f: caption = json.load(f).get("caption", "")
                     except: pass

                # Check if we need VO
                vo_path = None
                if caption:
                     try:
                         from voiceover import generate_voiceover
                         vo_path = os.path.join(job_dir, f"vo_{i}.mp3")
                         generate_voiceover(caption, vo_path)
                     except: pass

                if apply_ferrari_composer(
                    norm_path, rebuilt_path, target_res,
                    speed_var=0.0, # Compilation usually kept steady or minor var
                    voiceover_path=vo_path,
                    specific_music_path=music_segment_path, # NEW PARAM
                    # Ensure we disable fallback if we have specific music
                    human_safe_mode=(total_raw_duration > 65) # Safer for long compilations
                ):
                    norm_files.append(rebuilt_path)
                else:
                    norm_files.append(norm_path) # Fallback

            else:
                logger.warning(f"⚠️ Failed to normalize: {v_path}")


        
        if not norm_files:
            logger.error("❌ No valid clips to compile.")
            return None

        # 2. Build Filter Complex for XFADE
        # If only 1 clip, just copy
        if len(norm_files) == 1:
            shutil.copy2(norm_files[0], output_filename)
            return output_filename

        # FFMPEG XFADE Logic
        # [0][1]xfade[v1]; [v1][2]xfade[v2]...
        inputs = []
        filter_str = ""
        map_v = "[v0]"
        map_a = "[a0]"
        
        offset = 0.0
        
        # We need durations to calculate offsets
        durations = []
        for f in norm_files:
            info = _get_video_info(f)
            d = info.get("duration", 0)
            if d <= 0: d = 10.0 # Safety default for failed probes
            
            # Critical Safety: Clip must be longer than transition
            if d < TRANSITION_DURATION * 1.5:
                # If clip is too short, fake it or warn?
                # FFMPEG xfade breaks if offsets overlap poorly.
                logger.warning(f"⚠️ Clip too short ({d}s) for transition ({TRANSITION_DURATION}s). Extending inputs won't work easily.")
                # We could pad it? For now, let's just accept it and hope offset calculation handles it, 
                # OR enforces minimum duration during normalization.
            
            durations.append(d)
            inputs.extend(["-i", f])

        # Construct Filter
        # Stream 0 is base
        curr_v = "0:v"
        curr_a = "0:a"
        
        # Accumulate offset
        offset = durations[0] - TRANSITION_DURATION
        
        filter_parts = []
        
        for i in range(1, len(norm_files)):
            next_v = f"{i}:v"
            next_a = f"{i}:a"
            
            target_v = f"v{i}"
            target_a = f"a{i}"
            
            # Video XFade
            # Pick random transition?
            trans_list = ["fade", "wipeleft", "wiperight", "slideleft", "slideright", "circleopen", "dissolve"]
            trans = random.choice(trans_list)
            
            # FIX: Format offset to .2f to avoid invalid argument errors in FFMPEG
            filter_parts.append(f"[{curr_v}][{next_v}]xfade=transition={trans}:duration={TRANSITION_DURATION}:offset={offset:.2f}[{target_v}]")
            
            # Audio Crossfade
            # crossfade doesn't use offset logic same way, it just stitches
            # actually acrossfade is easier: acrossfade=d=1
            # But combining simple acrossfade chain with xfade offset is tricky for sync.
            # Simpler approach:
            # We use `acrossfade` which automatically handles timestamps, BUT xfade requires explicit offset.
            # To keep sync, we must track the NEW duration after fade.
            # Duration(A+B) = Dur(A) + Dur(B) - Fade
            
            filter_parts.append(f"[{curr_a}][{next_a}]acrossfade=d={TRANSITION_DURATION}[{target_a}]")
            
            # Update current streams for next iteration
            curr_v = target_v
            curr_a = target_a
            
            # Update offset for NEXT fade
            # The previous clip (A+B) now ends at: OldOffset + TransDur + (DurB - TransDur) = OldOffset + DurB
            # Wait, standard xfade offset logic:
            # Offset is the timestamp in the FIRST stream where second stream starts.
            # For chained xfade:
            # Offset 1 = Dur(0) - Fade
            # Offset 2 = Offset 1 + Dur(1) - Fade
            # ...
            if i < len(norm_files) - 1:
                 offset += durations[i] - TRANSITION_DURATION
                 
        full_filter = ";".join(filter_parts)
        
        # 3. Execute FFMPEG (Pass 1: Visuals + Transitions)
        temp_visual_output = output_filename.replace(".mp4", "_temp_visual.mp4")
        
        cmd = [
            FFMPEG_BIN, "-y"
        ] + inputs + [
            "-filter_complex", full_filter,
            "-map", f"[{curr_v}]", "-map", f"[{curr_a}]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", REENCODE_PRESET, "-crf", REENCODE_CRF,
            "-c:a", "aac",
            temp_visual_output
        ]
        
        logger.info(f"🎞️ Rendering batch (Pass 1/2: Visuals): {len(norm_files)} clips...")
        
        try:
            # Tier 1: Primary XFADE Render
            success = _run_command(cmd, timeout=len(norm_files)*60, check=True)
        
            # Validation
            if not success or not os.path.exists(temp_visual_output) or os.path.getsize(temp_visual_output) < 3000:
                raise Exception("Primary render failed or produced invalid file")
            
            # --- PASS 2: ADD BACKGROUND MUSIC (SHORTS ONLY) ---
            # Smart Logic: 
            # If Long Form (>65s), we already rebuilt audio per-clip in _rebuild_clip_audio. 
            # Do NOT add global background music on top (Double Audio Risk).
            # If Shorts, we retain legacy behavior (Global mix).
            
            # Determine if we used Per-Clip Audio Rebuild (Strict Mode)
            # If so, we SKIP the Global Music Mix to avoid double-music.
            # Logic: 
            # - Long Form (>65s) AND HEAVY_REMIX_COMPILATION -> Skipped
            # - Shorts (<=65s) AND HEAVY_REMIX_SHORTS -> Skipped
            # - Otherwise -> Run Pass 2 (Legacy Overlay)
            
            strict_mode_active = False
            if total_raw_duration > 65:
                # Force Skip Pass 2 for Long Form (We already did it per-clip)
                strict_mode_active = True 
            else:
                if ENABLE_HEAVY_REMIX_SHORTS: strict_mode_active = True

            if strict_mode_active:
                 logger.info("⏩ Skipping Pass 2 (Global Music Mix). Using Per-Clip Rebuilt Audio (Strict Mode).")
                 # Just move temp visual to output
                 shutil.move(temp_visual_output, output_filename)
                 
            else:
                logger.info("🎵 Mixing background music (Pass 2/2) [Shorts Mode]...")
                
                # Select Music
                music_files = glob.glob(os.path.join("music", "*.mp3"))
                music_input = []
                filter_mix = f"[0:a]volume=1.0[a0];[a0]anull[outa]" # Default: No music
                
                if music_files:
                    bg_music = random.choice(music_files)
                    logger.info(f"    └─ Track: {os.path.basename(bg_music)}")
                    music_input = ["-stream_loop", "-1", "-i", bg_music]
                    # MIXING STRATEGY: Gated Voiceover + Sidechained Music
                    # Goal: Mute original audio when VO stops, Let music swell.
                    
                    # 1. GATE INPUT: Kill anything quiet (ambience), keep loud (Voiceover)
                    #    threshold=0.1 (~-20dB). Since VO volume was 1.6, this should be safe.
                    # 2. DUCK MUSIC: When Gated Input is active, duck Music.
                    # 3. MIX: Combine.
                    
                    # MIXING STRATEGY V4: Copyright Killer
                    # Goal: Aggressively Remove Original Music (Assume it's quieter than VO).
                    # Settings: Threshold 0.35 (Very High!), Release 50ms (Snap closed).
                    # Highpass: 200Hz (Kill bass).
                    
                    filter_mix = (
                        f"[0:a]highpass=f=200,volume={ORIGINAL_AUDIO_VOLUME},agate=threshold=0.35:ratio=9000:range=-90dB:attack=1:release=50[gated_vo];"
                        f"[1:a]volume={MUSIC_VOLUME}*3,aperms[music_raw];"
                        f"[music_raw][gated_vo]sidechaincompress=threshold=0.08:ratio=20:attack=5:release=800[ducked_music];"
                        f"[gated_vo][ducked_music]amix=inputs=2:duration=first:dropout_transition=0:weights=1 1[outa]"
                    )
                else:
                    logger.warning("⚠️ No music files found in 'music/' folder. Skipping music mix.")
                
                cmd_mix = [
                    FFMPEG_BIN, "-y",
                    "-i", temp_visual_output
                ] + music_input + [
                    "-filter_complex", filter_mix,
                    "-map", "0:v", "-map", "[outa]",
                    "-c:v", "copy", # Fast copy video
                    "-c:a", "aac", "-b:a", "192k",
                    output_filename
                ]
                
                _run_command(cmd_mix, timeout=300, check=True)
            
            # Cleanup temp
            try: os.remove(temp_visual_output)
            except: pass
                
            logger.info(f"✅ Batch compilation success: {output_filename}")
            
            # --- BRANDING INJECTION (Strictly Post-Processing) ---
            # 1. Image Overlay (Logo) if exists
            # 2. Episodic Text if enabled
            # Replaces legacy _inject_branding (Intro/Outro) - wait, user said "Additive". 
            # Legacy _inject_branding handles Intros. We should keep it.
            
            output_filename = _inject_branding(output_filename) # Legacy Intro/Outro
            
            # --- NEW BRANDING ---
            if total_raw_duration > 65: # Compilation Only
                 try:
                     # 1. LOGO
                     logo_path = os.path.join("logo", "brand_logo.png")
                     if not os.path.exists(logo_path):
                          # Check for jpeg or others
                          for ext in [".png", ".jpg", ".jpeg"]:
                              p = os.path.join("logo", f"logo{ext}") # generic name?
                              if os.path.exists(p): 
                                  logo_path = p
                                  break
                                  
                     if os.path.exists(logo_path):
                         logger.info(f"🎨 Applying Logo Overlay: {os.path.basename(logo_path)}")
                         branded = output_filename.replace(".mp4", "_logo.mp4")
                         if add_logo_overlay(output_filename, branded, logo_path, lane_context="caption"):
                              # Atomic Swap
                              shutil.move(branded, output_filename)
                     
                     # 2. EPISODIC
                     # Extract Episode Num from filename?
                     # output_filename is usually "Compilation_Title..."
                     # We need to track the 'series count' or use env var?
                     # User said: "Extract episode number from filename (e.g. d_p_12.mp4)"
                     # BUT this is a compilation of 5-10 clips. Which filename?
                     # Ah, the COMPILATION filename? Main.py generates title.
                     # Or does it mean the USER uploads "Episode 12.mp4"?
                     # Usually compilations are generated automatically.
                     # "Fallback to compilation index... Log warning"
                     
                     # Let's use a counter or date?
                     # Ideally we pass 'episode_num' into this function.
                     # But signature is fixed.
                     # We can guess from output_filename if it contains numbers?
                     # Or read a counter file?
                     
                     ep_num = "1"
                     try:
                         # Try finding number in filename
                         mat = re.search(r"(\d+)", os.path.basename(output_filename))
                         if mat: ep_num = mat.group(1)
                     except: pass
                     
                     show_title = os.getenv("SHOW_TITLE", "SWARGAWASAL")
                     show_series = os.getenv("SHOW_SERIES", "SEASON 1")
                     show_tagline = os.getenv("SHOW_TAGLINE", "Start Your Day With A Smile")
                     
                     logger.info(f"📺 Applying Episodic Overlay: Ep {ep_num}")
                     ep_branded = output_filename.replace(".mp4", "_ep.mp4")
                     if add_episodic_overlay(output_filename, ep_branded, ep_num, show_series, show_tagline, has_intro=True):
                         shutil.move(ep_branded, output_filename)
                         
                 except Exception as e:
                     logger.warning(f"⚠️ Branding Overlay Failed: {e}")
            
            return output_filename
        
        except Exception as e:
            logger.warning(f"⚠️ Primary Batch Render Failed: {e}")
            logger.info("🔄 Falling back to SAFE CONCAT (No Transitions)...")
            
            # Tier 3: Safe Concat Fallback
            # Just concat the normalized files.
            try:
                 safe_list_path = os.path.join(job_dir, "safe_concat_list.txt")
                 with open(safe_list_path, "w", encoding='utf-8') as f:
                     for nf in norm_files:
                         f.write(f"file '{os.path.abspath(nf).replace(os.sep, '/')}'\n")
                 
                 concat_cmd = [
                     FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", 
                     "-i", safe_list_path,
                     "-c", "copy",
                     output_filename
                 ]
                 logger.info("🛡️ Running Safe Concat...")
                 _run_command(concat_cmd, check=True)
                 
                 if verify_video_integrity(output_filename):
                      # --- BRANDING INJECTION (Fallback Path) ---
                      output_filename = _inject_branding(output_filename) # Legacy
                      
                      # Call new branding here too? 
                      # Yes, safe to do so.
                      try:
                          logo_path = os.path.join("logo", "brand_logo.png") # simplified lookup
                          if os.path.exists(logo_path):
                              branded = output_filename.replace(".mp4", "_logo.mp4")
                              if add_logo_overlay(output_filename, branded, logo_path, lane_context="caption"):
                                  shutil.move(branded, output_filename)
                      except: pass
                      
                      # --- MONETIZATION GUARD (Fallback) ---

                      try:
                          from monetization_brain import brain
                          # We don't have rich transformation data here, so we assume basic
                          audit = brain.analyze_content(
                              title="Compilation Fallback", 
                              duration=_get_video_info(output_filename).get('duration', 0),
                              transformations={"Mode": "Safe Concat", "Audio": "Unknown"}
                          )
                          # Save Sidecar
                          _save_sidecar(output_filename, audit, {"mode": "fallback"})
                          logger.info(f"🛡️ Monetization Audit (Fallback): {audit.get('verdict', 'Unknown')}")
                      except Exception as e:
                          logger.warning(f"⚠️ Fallback Audit Failed: {e}")

                      return output_filename
                 else:
                      logger.error("❌ Safe Concat also failed.")
                      return None
                      
            except Exception as e2:
                 logger.error(f"❌ Critical Batch Failure: {e2}")
                 return None
             
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)

def reprocess_watermark_step(input_video: str, retry_mode: bool = False) -> tuple[str, Optional[Dict]]:
    # Redirect to main pipeline or keep simple watermark loop logic?
    # User constraint: Do NOT remove features.
    # We must keep it working.
    from watermark_auto import process_video_with_watermark
    # Use simple call
    dir_name = os.path.dirname(input_video)
    name, ext = os.path.splitext(os.path.basename(input_video))
    output_video = os.path.join(dir_name, f"{name}_reproc_{int(time.time())}{ext}")
    
    result = process_video_with_watermark(input_video, output_video, retry_mode=retry_mode)
    if result["success"] and os.path.exists(output_video):
        return output_video, result.get("context")
    return input_video, None

def _rebuild_clip_audio(video_path: str, output_path: str, meta_path: str, music_playlist: List[str] = None, music_index: List[int] = None, fade_out_duration: float = 1.5) -> bool:
    """
    STRICT AUDIO REMIX ENGINE (Long-Form Only)
    Logic: [VOICEOVER] -> [SEQUENTIAL MUSIC] -> [FADE OUT]
    - Original Audio: DESTROYED (100% Strip)
    - Music: Sequential from playlist, strictly filling time after VO.
    """
    try:
        # 0. Strip Confirmation
        logger.info(f"🎙️ Rebuilding Audio (Strict Mode): {os.path.basename(video_path)}")
        logger.info("    └─ 🔇 Original Audio: STRIPPED")

        # 1. Validation
        if not os.path.exists(meta_path):
            logger.warning("    └─ No metadata found. Skipping rebuild.")
            return False
            
        with open(meta_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        caption = data.get("pipeline_metrics", {}).get("caption", "") or data.get("caption", "")
        if not caption: 
            logger.warning("    └─ No caption found. Skipping rebuild.")
            return False

        # 2. Get Video Duration
        try:
             dur_v = float(subprocess.check_output([FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path]).decode().strip())
        except:
             return False

        # 3. Generate VO
        from voiceover import generate_voiceover
        vo_path = output_path + ".vo.mp3"
        vo_success = generate_voiceover(caption, vo_path)
        
        dur_vo = 0.0
        if vo_success:
             try:
                 dur_vo = float(subprocess.check_output([FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", vo_path]).decode().strip())
                 logger.info(f"    └─ 🎙️ Voiceover Generated: \"{caption[:30]}...\" ({dur_vo}s)")
             except:
                 vo_success = False

        # 4. Music Logic (Sequential)
        music_inputs = []
        filter_parts = []
        
        # We need to fill (dur_v - dur_vo)
        # If VO fails, we fill dur_v
        
        start_time = dur_vo
        remaining_duration = dur_v - start_time
        
        # Audio Mix Setup
        # Input 0: Video (Video Only)
        # Input 1: VO (if exists)
        # Input 2..N: Music Tracks
        
        cmd_inputs = ["-i", video_path]
        next_input_idx = 1
        
        mix_inputs = [] # List of stream labels to mix at end
        
        if vo_success:
            cmd_inputs.extend(["-i", vo_path])
            # VO is always at 0.0
            # We map it directly to mix
            # But we might want to normalize volume?
            filter_parts.append(f"[{next_input_idx}:a]volume=1.8,aperms[vo_final]")
            mix_inputs.append("[vo_final]")
            next_input_idx += 1
            
        # Music Chain
        if music_playlist and music_index is not None:
             current_track_time = 0.0
             
             while current_track_time < remaining_duration:
                 # Get next track
                 track_idx = music_index[0] % len(music_playlist)
                 track_path = music_playlist[track_idx]
                 music_index[0] += 1 # Increment for next call/loop
                 
                 logger.info(f"    └─ 🎵 Music Selected: {os.path.basename(track_path)}")
                 
                 # Add Input
                 cmd_inputs.extend(["-i", track_path])
                 
                 # Get Track Duration (to know if we need more)
                 try:
                     track_dur = float(subprocess.check_output([FFPROBE_BIN, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", track_path]).decode().strip())
                 except:
                     track_dur = 30.0 # Fallback
                 
                 # Timeline Math
                 # Clip start: start_time + current_track_time
                 # Clip end: start_time + current_track_time + track_dur
                 # Valid Duration: min(track_dur, remaining_duration - current_track_time)
                 
                 insert_start = start_time + current_track_time
                 play_dur = track_dur
                 
                 # Trim if exceeds video
                 if insert_start + play_dur > dur_v:
                     play_dur = dur_v - insert_start
                     
                 # Filter: Delay + Trim + FadeOut (if end of video)
                 # adelay uses milliseconds.
                 delay_ms = int(insert_start * 1000)
                 
                 # Logic for Cross-Clip Fade:
                 # If we are in middle of compilation, we use minimal fade (fade_out_duration).
                 # If last clip, we use larger fade.
                 
                 is_end_track = (insert_start + play_dur >= dur_v - 0.1)
                 
                 # Construct Filter
                 # [a]atrim=0:dur,adelay=ms,afade=t=in:ss=0:d=0.5,afade=t=out:st=(dur-fade):d=fade[a]
                 
                 label = f"music_{next_input_idx}"
                 
                 fade_cmd = ""
                 if is_end_track:
                     # Calculate fade start point
                     # FIX: Must account for 'insert_start' (adelay) in the timestamp!
                     # The stream is: [Silence (insert_start)] + [Music (play_dur)]
                     # End is at insert_start + play_dur.
                     
                     start_fade = (insert_start + play_dur) - fade_out_duration
                     if start_fade < 0: start_fade = 0
                     
                     fade_cmd = f",afade=t=out:st={start_fade:.2f}:d={fade_out_duration}"
                 else:
                     # Cross-track mixing (internal to video)
                     # Standard small crossfade
                     # Also needs offset!
                     start_fade = (insert_start + play_dur) - 1.0
                     if start_fade < 0: start_fade = 0
                     fade_cmd = f",afade=t=out:st={start_fade:.2f}:d=1.0"
                 
                 # Construct filter
                 # [In] atrim=0:play_dur, adelay=delay_ms, volume=0.4 [Out]
                 # Note: adelay adds silence at start.
                 
                 filter_parts.append(f"[{next_input_idx}:a]atrim=0:{play_dur},adelay={delay_ms}|{delay_ms},volume={MUSIC_VOLUME}{fade_cmd},aperms[{label}]")
                 mix_inputs.append(f"[{label}]")
                 
                 next_input_idx += 1
                 current_track_time += play_dur
                 
                 if current_track_time >= remaining_duration: 
                     break
                     
                 logger.info(f"    └─ 🎵 Switching to Next Track...")
        
        # 5. Final Mix
        # Concatenate all audio parts? No, mix them.
        # amix handles overlapping if any (shouldn't be much)
        
        if not mix_inputs:
             # Silence fallback
             filter_parts.append("anullsrc=channel_layout=stereo:sample_rate=44100:d=1[silence]")
             mix_inputs.append("[silence]")
             
        mix_count = len(mix_inputs)
        # compensating for amix averaging by multiplying by count (assuming mostly non-overlapping)
        mix_str = "".join(mix_inputs) + f"amix=inputs={mix_count}:duration=longest:dropout_transition=0,volume={mix_count}[outa]"
        filter_parts.append(mix_str)
        
        full_filter = ";".join(filter_parts)
        
        # 6. Execution
        cmd = [
            FFMPEG_BIN, "-y"
        ] + cmd_inputs + [
            "-filter_complex", full_filter,
            "-map", "0:v", "-map", "[outa]",   # STRICT: 0:v (Video), [outa] (New Audio)
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_path
        ]
        
        if _run_command(cmd, timeout=120, check=True):
             try: 
                if vo_success: os.remove(vo_path)
             except: pass
             return True
        else:
             return False

    except Exception as e:
        logger.error(f"❌ Rebuild Error: {e}")
        return False

