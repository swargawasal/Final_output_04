"""
Text Overlay Module (Hardened Production Grade)
Handles robust text overlay with Font Auto-Healing via Official Zip, ASS Fallback, and Crash Safety.

Capabilities:
1. Auto-downloads authoritative font (Inter v4.0 Zip).
2. Extracts and verifies font file integrity (>50KB).
3. Falls back to subtitle overlay (.ass) if drawtext fails or unicode detected.
4. Sanitizes all text inputs.
5. Non-blocking failure model (returns False instead of crashing).

STRICT AUDIT COMPLIANT: Global State Fix, Atomic Ops, Conservative Width.
"""

import os
import subprocess
import logging
import shutil
import textwrap
import requests
import zipfile
import io
import time
import threading
import json
from typing import Optional, Dict, Any
import tempfile

logger = logging.getLogger("text_overlay")

FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FONT_ZIP_URL = "https://github.com/rsms/inter/releases/download/v4.0/Inter-4.0.zip"
LOCAL_FONT_DIR = os.path.join("assets", "fonts")
LOCAL_FONT_PATH = os.path.join(LOCAL_FONT_DIR, "Inter-Bold.ttf")

# 1. Configurable Env Vars
FONT_MIN_SIZE_BYTES = int(os.getenv("FONT_MIN_SIZE_BYTES", 50 * 1024))
FONT_DOWNLOAD_TIMEOUT_SECS = int(os.getenv("FONT_DOWNLOAD_TIMEOUT_SECS", 30))
FONT_DOWNLOAD_RETRIES = int(os.getenv("FONT_DOWNLOAD_RETRIES", 2))
FONT_AUTO_DOWNLOAD_BACKGROUND = os.getenv("FONT_AUTO_DOWNLOAD_BACKGROUND", "yes").lower() == "yes"
ASS_PLAYRES_X = int(os.getenv("ASS_PLAYRES_X", 1080))
ASS_PLAYRES_Y = int(os.getenv("ASS_PLAYRES_Y", 1920))
DEBUG_JSON = os.getenv("DEBUG_JSON", "0") == "1"
TEXT_MAX_CHARS = int(os.getenv("TEXT_MAX_CHARS", 220))

# Text Lane Configuration (Vertical Video Optimized)
TEXT_LANES = {
    "caption": 0.72,   # AI Captions (MOVED UP: Safe Zone for Bottom Overlay)
    "fixed": 0.88,     # Fixed Branding/Tagline (Bottom 12%)
    "top": 0.08,       # Top Warning/Info
    "center": 0.50     # Dead Center
}
LANE_PADDING = 0.04

class TextOverlay:
    def __init__(self):
        # Instance-scoped state to prevent global poisoning
        self._drawtext_supported: Optional[bool] = None
        self._font_checked: bool = False
        self._drawtext_failed_once: bool = False
        self._last_result_meta: Dict[str, Any] = {}
        self._last_debug_info: Dict[str, Any] = {}
        
        # 17. Non-Blocking Font Download
        if self._validate_font_file(LOCAL_FONT_PATH):
             self._font_checked = True
             self._check_drawtext_support()
        else:
             if FONT_AUTO_DOWNLOAD_BACKGROUND:
                 t = threading.Thread(target=self._ensure_font_thread, daemon=True)
                 t.start()
                 # We don't block; fallback to ASS until ready
             else:
                 self._ensure_font_thread()
                 self._check_drawtext_support()

    def _ensure_font(self):
        """Deprecated: Use _ensure_font_thread internal logic."""
        self._ensure_font_thread()

    def _ensure_font_thread(self):
        """Auto-heals missing font by downloading and extracting the official Zip (Atomic & Robust)."""
        if self._font_checked: return

        # Double check in thread
        if self._validate_font_file(LOCAL_FONT_PATH):
            self._font_checked = True
            return

        os.makedirs(LOCAL_FONT_DIR, exist_ok=True)
        # Use temp file for atomic swap
        fd, temp_zip = tempfile.mkstemp(suffix=".zip", dir=LOCAL_FONT_DIR)
        os.close(fd)
        temp_ttf = os.path.join(LOCAL_FONT_DIR, f"tmp_{int(time.time())}.ttf")

        for attempt in range(FONT_DOWNLOAD_RETRIES + 1):
            try:
                logger.info(f"⬇️ Downloading font (Attempt {attempt+1})...")
                response = requests.get(FONT_ZIP_URL, timeout=FONT_DOWNLOAD_TIMEOUT_SECS)
                response.raise_for_status()
                
                with open(temp_zip, "wb") as f:
                    f.write(response.content)

                found = False
                with zipfile.ZipFile(temp_zip) as z:
                    target_file = None
                    for name in z.namelist():
                        if name.endswith("Inter-Bold.ttf") and "Variable" not in name:
                            target_file = name
                            break
                    
                    if target_file:
                        with z.open(target_file) as source, open(temp_ttf, "wb") as target:
                            shutil.copyfileobj(source, target)
                        
                        # Atomic Move
                        if self._validate_font_file(temp_ttf):
                            # On Windows, need to remove dest first
                            if os.path.exists(LOCAL_FONT_PATH):
                                try: os.remove(LOCAL_FONT_PATH) 
                                except: pass
                            os.replace(temp_ttf, LOCAL_FONT_PATH)
                            self._font_checked = True
                            logger.info("✅ Font installed successfully.")
                            found = True
                            break # Success
                        else:
                            logger.error("❌ Downloaded font validation failed.")
                    else:
                        logger.error("❌ Inter-Bold.ttf not found in ZIP.")
                
                if found: break
                
            except Exception as e:
                logger.warning(f"⚠️ Font download attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (2 ** attempt)) # Exponential backoff
            finally:
                # Cleanup temps
                for p in [temp_zip, temp_ttf]:
                    if os.path.exists(p):
                        try: os.remove(p)
                        except: pass
        
        # Final check
        if not self._font_checked:
            logger.error("❌ Failed to install font after retries. Subtitles fallback enabled.")

    def _validate_font_file(self, path: str) -> bool:
        if not os.path.exists(path): return False
        try:
            if os.path.getsize(path) < FONT_MIN_SIZE_BYTES: return False
            # Header check
            with open(path, "rb") as f:
                header = f.read(4)
                # TrueType (00010000) or OTC/TTF (ttcf)
                if header == b'\x00\x01\x00\x00' or header == b'ttcf':
                    return True
            return False
        except:
            return False

    def _check_drawtext_support(self):
        """Checks if installed FFmpeg supports drawtext filter."""
        if self._drawtext_supported is not None:
            return

        try:
            # 4. Drawtext Support Check with Timeout
            result = subprocess.run(
                [FFMPEG_BIN, "-filters"], 
                capture_output=True, 
                text=True,
                timeout=6 # Strict Timeout
            )
            # Robust stdout parsing
            output = result.stdout.lower()
            self._drawtext_supported = " drawtext " in output or "\ndrawtext " in output
            
            if not self._drawtext_supported:
                logger.warning("⚠️ FFmpeg 'drawtext' filter NOT found. Fallback mode enabled.")
        except subprocess.TimeoutExpired:
            logger.warning("⚠️ FFmpeg filter check timed out. Assuming broken.")
            self._drawtext_supported = False
        except Exception:
            logger.warning("⚠️ Could not verify FFmpeg filters. Assuming broken.")
            self._drawtext_supported = False

    def _is_safe_ascii(self, text: str) -> bool:
        """Strict check for drawtext safety. ONLY printable ASCII allowed."""
        if not text: return True
        try:
            # Must be ASCII
            text.encode('ascii')
            # detailed check for control chars
            for char in text:
                if not (32 <= ord(char) <= 126 or char == '\n'):
                    return False
            return True
        except UnicodeEncodeError:
            return False

    def _escape_drawtext(self, text: str) -> str:
        """Strict escaping for FFmpeg drawtext."""
        if not text: return ""
        # Filter logic special chars
        text = text.replace("\\", "\\\\")
        text = text.replace(":", "\\:")
        text = text.replace("'", "'\\''") 
        text = text.replace("%", "\\%")
        # text = text.replace("\n", "\\n") # Do NOT double escape here if using newlines for lines?
        # Actually filtergraph expects literal newlines or specific sequences.
        # We will split lines manually, so no need to handle newlines inside the string usually.
        # But if we did:
        text = text.replace("\n", " ") # Collapse for safety within a single line
        
        text = text.replace("[", "\\[").replace("]", "\\]")
        
        # Collapse whitespace
        text = " ".join(text.split())
        
        # Truncate
        if len(text) > TEXT_MAX_CHARS:
             text = text[:TEXT_MAX_CHARS] + "..."
             
        return text

    def _escape_ass(self, text: str) -> str:
        """Escaping for ASS subtitles."""
        if not text: return ""
        text = text.replace("{", "\\{").replace("}", "\\}")
        text = text.replace("\n", "\\N") 
        return text

    def _wrap_text(self, text: str, max_chars: int = 26) -> str:
        if not text: return ""
        text = text.replace("\r", "").strip()
        if len(text) <= max_chars: return text
        return textwrap.fill(text, width=max_chars, break_long_words=False, break_on_hyphens=False)

    def _create_ass_file(self, text: str, lane: str) -> str:
        """Generates a temporary .ass subtitle file (Atomic Write)."""
        filename = f"overlay_{os.getpid()}_{int(time.time()*1000)}.ass"
        tmp_dir = os.path.join("temp", "ass")
        os.makedirs(tmp_dir, exist_ok=True)
        
        ass_path = os.path.join(tmp_dir, filename)
        
        # Use temp file for writing
        fd, temp_write_path = tempfile.mkstemp(dir=tmp_dir, text=True)
        os.close(fd)
        
        # Alignment: 2=Bottom Center, 8=Top Center, 5=Middle Center
        alignment = 2
        
        # Calc MarginV based on Lane Percentage
        pct = TEXT_LANES.get(lane, TEXT_LANES["caption"])
        
        if lane == "top":
            alignment = 8
            margin_v = int(ASS_PLAYRES_Y * pct)
        elif lane == "center":
            alignment = 5
            margin_v = 0
        else:
             # Bottom Logic (Caption/Fixed)
             # MarginV = Distance from Bottom
             margin_v = int(ASS_PLAYRES_Y * (1.0 - pct))

        escaped_text = self._escape_ass(text)

        ass_content = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {ASS_PLAYRES_X}
PlayResY: {ASS_PLAYRES_Y}
WrapStyle: 1

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,0,{alignment},20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,9:59:59.00,Default,,0,0,0,,{escaped_text}
"""
        with open(temp_write_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        
        # Atomic Move
        if os.path.exists(ass_path):
             try: os.remove(ass_path)
             except: pass
        os.replace(temp_write_path, ass_path)
        
        return ass_path

    def add_overlay(self, video_path, output_path, text, lane="caption", size=60):
        """
        Main entry point with Strict Fallback Logic and Lanes.
        """
        if not text or not video_path or not os.path.exists(video_path):
            return False

        if lane == "caption":
             wrapped_text = self._wrap_text(text, max_chars=22) # User Req: ~22 chars
        else:
             wrapped_text = self._wrap_text(text, max_chars=24)
        
        # 11. Font Size & Overflow Protection
        line_count = wrapped_text.count("\n") + 1
        
        if line_count == 2:
            size = int(size * 0.85)
        elif line_count >= 3:
            size = int(size * 0.70)
            
        longest_line = max([len(line) for line in wrapped_text.split('\n')]) if wrapped_text else 0
        
        # Conservative Width Estimate (0.7 factor)
        estimated_width = longest_line * (size * 0.7)
        max_allowed_width = ASS_PLAYRES_X * 0.9
        
        while estimated_width > max_allowed_width and size > 18:
             size = int(size * 0.95)
             estimated_width = longest_line * (size * 0.7)
        
        size = max(18, min(size, 300))

        if lane not in TEXT_LANES:
            logger.warning(f"⚠️ Unknown text lane '{lane}', defaulting to 'caption'")
            lane = "caption"

        # Decision Tree
        use_drawtext = True
        start_method = "DRAWTEXT"
        reason = "optimal"
        
        if self._drawtext_failed_once:
             use_drawtext = False
             reason = "previous_failure"
        elif not self._drawtext_supported:
             use_drawtext = False
             reason = "drawtext_unavailable"
        elif not self._font_checked and not os.path.exists(LOCAL_FONT_PATH):
             use_drawtext = False
             reason = "font_missing"
        elif not self._is_safe_ascii(text):  # STRICT ALLOWLIST
             use_drawtext = False
             reason = "complex_chars_detected"
        elif size < 20:
             use_drawtext = False
             reason = "text_too_small"

        if not use_drawtext:
             start_method = "SUBTITLES"
        
        logger.info("🧾 [TEXT OVERLAY] Starting")
        logger.info(f"    ├─ lane: {lane}")
        logger.info(f"    ├─ text_len: {len(text)}")
        logger.info(f"    ├─ size: {size}")
        logger.info(f"    └─ method: {start_method} ({reason})")

        self._last_result_meta = {
            "method": start_method.lower(),
            "font_used": LOCAL_FONT_PATH if use_drawtext else None,
            "text_len": len(text),
            "size": size,
            "lane": lane
        }

        if use_drawtext:
            success = self._apply_drawtext(video_path, output_path, wrapped_text, lane, size)
            if success:
                logger.info("✔️ [TEXT OVERLAY] Applied successfully")
                logger.info("    ├─ method: DRAWTEXT")
                logger.info("    └─ lane: " + lane)
                return True
            else:
                self._drawtext_failed_once = True # Only fails THIS instance
                logger.warning("❌ [TEXT OVERLAY] Drawtext failed")
                logger.warning("    ├─ reason: ffmpeg_error")
                logger.warning("    └─ action: fallback_to_subtitles")
                self._last_result_meta["fallback"] = True
                return self._apply_ass(video_path, output_path, wrapped_text, lane)
        else:
            success = self._apply_ass(video_path, output_path, wrapped_text, lane)
            if success: 
                logger.info("✔️ [TEXT OVERLAY] Applied successfully")
                logger.info("    ├─ method: SUBTITLES")
                logger.info("    └─ lane: " + lane)
            return success

    def _safe_run_overlay(self, method_name, cmd):
        """Wrapper for safe subprocess execution."""
        try:
            # Timeout 120s
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=120)
            
            if DEBUG_JSON:
                 self._last_debug_info = {
                     "method": method_name,
                     "cmd": " ".join(cmd[:10]) + "...",
                     "result": True
                 }
            return True
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode() if e.stderr else 'Unknown Error'
            logger.error(f"{method_name} failed: {err_msg[:300]}")
            if DEBUG_JSON:
                 self._last_debug_info = {
                     "method": method_name,
                     "cmd": " ".join(cmd[:10]) + "...",
                     "result": False,
                     "error": err_msg
                 }
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"{method_name} timed out!")
            return False
        except Exception as e:
            logger.error(f"{method_name} crashed: {e}")
            return False

    def _apply_drawtext(self, video_path, output_path, text, lane, size):
        font_path = os.path.abspath(LOCAL_FONT_PATH).replace("\\", "/").replace(":", "\\:")
        
        lines = []
        if lane == "caption":
             clean_text = text.replace("\r", "")
             pre_lines = clean_text.split('\n')
             for line in pre_lines:
                 wrapped = textwrap.wrap(line, width=32, break_long_words=False)
                 lines.extend(wrapped)
        else:
             lines = text.split('\n')
             
        if len(lines) > 4: lines = lines[:4]
        
        filters = []
        
        if lane == "fixed":
             # Brand logic (Bottom Anchor)
             fixed_size = int(size * 0.9)
             # Explicitly anchor at bottom 10%
             fixed_y = "h*0.90" 
             for line in lines:
                 safe_line = self._escape_drawtext(line)
                 dt = (f"drawtext=fontfile='{font_path}':text='{safe_line}':fontsize={fixed_size}:"
                       f"fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y={fixed_y}")
                 filters.append(dt)
        
        elif lane == "caption":
             # NEW: Bottom-Up Anchoring (Grow Upwards)
             # Goal: Sit approx 20px above Branding (which is at h*0.90 or h*0.88).
             # Let's define Branding Top ~ h*0.88.
             # Anchor Point = h*0.88 - 20px.
             
             line_height = int(size * 1.25)
             total_lines = len(lines)
             
             for i, line in enumerate(lines):
                 safe_line = self._escape_drawtext(line)
                 if not safe_line: continue
                 
                 # Logic: Y = Anchor - (Lines_Left_Below * Height)
                 # i=0 (Top): Needs to be highest.
                 # i=1 (Bot): Needs to be lowest.
                 # Distance from Anchor = (Total - i) * Height
                 
                 # Using expression for ffmpeg
                 # h*0.86 is roughly top of Branding area
                 y_expr = f"(h*0.86) - ({total_lines - i} * {line_height}) - 20"
                 
                 dt = (f"drawtext=fontfile='{font_path}':text='{safe_line}':fontsize={size}:"
                       f"fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y={y_expr}")
                 filters.append(dt)
                 
        else:
             # Standard Top-Down for other lanes (top/center)
             # Fallback to legacy logic
             line_height = int(size * 1.25)
             for i, line in enumerate(lines):
                 safe_line = self._escape_drawtext(line)
                 if not safe_line: continue
                 y_expr = f"h*{TEXT_LANES.get(lane, 0.5)} + ({i} * {line_height})"
                 dt = (f"drawtext=fontfile='{font_path}':text='{safe_line}':fontsize={size}:"
                       f"fontcolor=white:borderw=2:bordercolor=black:x=(w-text_w)/2:y={y_expr}")
                 filters.append(dt)

        if not filters: return True

        complex_filter = ",".join(filters)
        cmd = [
            FFMPEG_BIN, "-y", "-i", video_path,
            "-vf", complex_filter,
            "-c:v", "libx264",
            "-preset", os.getenv("REENCODE_PRESET", "ultrafast"),
            "-crf", os.getenv("REENCODE_CRF", "23"),
            "-c:a", "copy",
            output_path
        ]
        
        return self._safe_run_overlay("Drawtext", cmd)

    def _apply_ass(self, video_path, output_path, text, lane):
        ass_file = None
        try:
            ass_file = self._create_ass_file(text, lane)
            safe_ass_path = os.path.abspath(ass_file).replace("\\", "/").replace(":", "\\:")
            vf_filter = f"subtitles='{safe_ass_path}'"

            cmd = [
                FFMPEG_BIN, "-y", "-i", video_path,
                "-vf", vf_filter,
                "-c:v", "libx264",
                "-preset", os.getenv("REENCODE_PRESET", "ultrafast"),
                "-crf", os.getenv("REENCODE_CRF", "23"),
                "-c:a", "copy",
                output_path
            ]
            
            return self._safe_run_overlay("ASS", cmd)
            
        except Exception as e:
            logger.error(f"ASS prep crashed: {e}")
            return False
        finally:
            # Safe cleanup after logic
            if ass_file and os.path.exists(ass_file):
                try: os.remove(ass_file) 
                except: pass

    def last_debug(self):
        return self._last_debug_info


    def add_logo_overlay(self, video_path: str, output_path: str, logo_path: str, lane_context: str = "caption") -> bool:
        """
        Adds a logo overlay to the video.
        
        Refinement Logic:
        - Position: Bottom-Left (Touched to edge or slightly offset)
        - Collision Avoidance: If 'caption' lane is active (bottom), move logo UP by 2% Y or keep padding.
        - Size: 7.4% of frame width (Fixed)
        """
        if not os.path.exists(logo_path): return False
        
        # Determine strict position
        # Standard: 10px from left, 10px from bottom (or h-h_logo-10)
        
        # If lane_context == 'caption' (User has captions at bottom):
        # We need to ensure we don't overlap. 
        # Captions usually start at h*0.88 approx.
        # Logo is small (~7%).
        # If we align bottom-left, we are usually fine horizontally, but vertically?
        # Captions are centered. Logo is Left.
        # Overlap risk is low UNLESS captions are very wide.
        # User requested: "If captions bottom: Move logo up by +2% Y"
        
        y_pos = "H-h-20" # Default: 20px from bottom
        
        if lane_context == "caption":
            # Move up slightly to clear the "bottom bar" feel
            y_pos = "H-h-50" # Push up 50px (~2.5%)
            
        # Scale filter: scale=iw*0.074:-1
        # Overlay filter: overlay=10:y_pos
        
        filter_str = (
            f"[1:v]scale=iw*0.074:-1[logo];"
            f"[0:v][logo]overlay=20:{y_pos}[out]"
        )
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", video_path, "-i", logo_path,
            "-filter_complex", filter_str,
            "-map", "[out]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "copy",
            output_path
        ]
        
        return self._safe_run_overlay("LogoOverlay", cmd)

    def add_episodic_overlay(self, video_path: str, output_path: str, episode_num: str, series_name: str = "SERIES", tagline: str = "TAGLINE", has_intro: bool = False) -> bool:
        """
        Adds episodic framing text (3 lines).
        
        Refinement Logic:
        - Fade In: 0.3s
        - Fade Out: 0.3s
        - Duration: 1.5s
        - Timing: Starts at 0s (or after intro offset if we knew it? assume 0s is start of *this* clip or *compilation*?) 
          "Appear only in: First 1.5s" -> implying t=0 to t=1.5.
          If `has_intro` is True, this logic might need adjustment if we are post-intro.
          But usually this function is called on the compiled file?
          Or on the first clip?
          Ideally on the compiled file.
        """
        if not self._font_checked and not os.path.exists(LOCAL_FONT_PATH):
             self._ensure_font_thread() # Try one last panic load
        
        font_path = os.path.abspath(LOCAL_FONT_PATH).replace("\\", "/").replace(":", "\\:")
        
        # Design:
        # Line 1: Series Name (Small, Spaced)
        # Line 2: Episode N (Large, Bold)
        # Line 3: Tagline (Medium, Italic/Color)
        
        # Timing
        start_t = 0.5
        dur = 3.5 # Total duration visible
        fade_dur = 0.3
        
        # If has_intro, usually intro is ~3-5s. 
        # But we are overlaying on the FINAL video?
        # "After intro" -> If we detect intro, we might shift start_t.
        # Let's assume caller handles offsets or we assume standard intro length if has_intro=True.
        # User said: "After intro (if intro exists)". 
        # We'll default to start_t = 4.0 if has_intro, else 0.5
        if has_intro:
             start_t = 4.5
        
        enable = f"between(t,{start_t},{start_t+dur})"
        
        # Alpha Fade
        # fade(t, start_t, fade_in) * fade_out(t, end, fade_out)
        # alpha='if(lt(t,0.5),0,if(lt(t,0.8),(t-0.5)/0.3,if(lt(t,3.5),1,if(lt(t,3.8),(3.8-t)/0.3,0))))'
        # Simpler: alpha=min(1,(t-start)/fade)*min(1,(end-t)/fade) implies linear
        alpha_expr = f"min(1,(t-{start_t})/{fade_dur})*min(1,({start_t+dur}-t)/{fade_dur})"
        
        # Center X, Y tiers
        # Series: Y=30%
        # Ep: Y=35%
        # Tag: Y=45%
        
        lines = []
        
        # 1. SERIES
        if series_name:
            t1 = self._escape_drawtext(series_name.upper())
            lines.append(f"drawtext=fontfile='{font_path}':text='{t1}':fontsize=40:fontcolor=white:x=(w-text_w)/2:y=h*0.35:alpha='{alpha_expr}':enable='{enable}'")
            
        # 2. EPISODE
        if episode_num:
            t2 = f"EPISODE {episode_num}"
            lines.append(f"drawtext=fontfile='{font_path}':text='{t2}':fontsize=90:fontcolor=yellow:borderw=3:bordercolor=black:x=(w-text_w)/2:y=h*0.40:alpha='{alpha_expr}':enable='{enable}'")

        # 3. TAGLINE
        if tagline:
            t3 = self._escape_drawtext(tagline)
            lines.append(f"drawtext=fontfile='{font_path}':text='{t3}':fontsize=50:fontcolor=white:x=(w-text_w)/2:y=h*0.48:alpha='{alpha_expr}':enable='{enable}'")
            
        if not lines: return False
        
        filter_str = ",".join(lines)
        
        cmd = [
            FFMPEG_BIN, "-y", "-i", video_path,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "copy",
            output_path
        ]
         
        return self._safe_run_overlay("EpisodicOverlay", cmd)

# Global Instance
overlay_engine = TextOverlay()

def apply_text_overlay_safe(input_path, output_path, text, lane="caption", size=60):
    return overlay_engine.add_overlay(input_path, output_path, text, lane, size)

def add_logo_overlay(video_path, output_path, logo_path, lane_context="caption"):
    return overlay_engine.add_logo_overlay(video_path, output_path, logo_path, lane_context)

def add_episodic_overlay(video_path, output_path, episode, series="SERIES", tagline="TAGLINE", has_intro=False):
    return overlay_engine.add_episodic_overlay(video_path, output_path, episode, series, tagline, has_intro)
