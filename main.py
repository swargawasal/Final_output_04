from dotenv import load_dotenv
load_dotenv()

import os
import glob
import logging
import asyncio
import shutil
import sys
import re
import time
import subprocess
import csv
import json
import io
import uuid
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
import community_promoter

# Configurable Constants with Safe Defaults

CLEANUP_POLICY = os.getenv("CLEANUP_POLICY", "delayed") # immediate, on_success, delayed
DEBUG_JSON = int(os.getenv("DEBUG_JSON", "0"))
NET_RETRY_COUNT = int(os.getenv("NET_RETRY_COUNT", "3"))
NET_BACKOFF_BASE = float(os.getenv("NET_BACKOFF_BASE", "2.0"))
LOCK_WAIT_SECS = int(os.getenv("LOCK_WAIT_SECS", "5"))
TELEGRAM_MAX_UPLOAD_MB = int(os.getenv("TELEGRAM_MAX_UPLOAD_MB", "50"))
SESSION_TTL_SECS = int(os.getenv("SESSION_TTL_SECS", "86400"))
THREAD_POOL_SIZE = int(os.getenv("THREAD_POOL_SIZE", "4"))


# Directory Setup
JOB_DIR = "jobs"
COMPILATIONS_DIR = "final_compilations"

os.makedirs(JOB_DIR, exist_ok=True)
os.makedirs(COMPILATIONS_DIR, exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("downloads", exist_ok=True) # Auto-Heal: Ensure downloads folder exists
os.makedirs("music", exist_ok=True) # Auto-Heal: Ensure music folder exists for Fallback Audio
os.makedirs("Original_audio", exist_ok=True) # Auto-Heal: Audio Pool Library
os.makedirs("remarks", exist_ok=True)
os.makedirs("logo", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Thread Pool for Heavy Tasks
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, CallbackQueryHandler, filters
from telegram.error import NetworkError, TimedOut



# Constants
ALLOWED_DOMAINS = ["instagram.com", "youtube.com", "youtu.be"]

# Logging Setup
# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/bot.log", encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# --- SMART LOGGING FILTER ---
class PollingFilter(logging.Filter):
    def filter(self, record):
        # Filter out "getUpdates" spam but allow other API calls
        return "getUpdates" not in record.getMessage()

# Apply filter to noisy libraries
# We allow INFO level but filter out the polling spam
for lib in ["httpx", "telegram", "apscheduler"]:
    l = logging.getLogger(lib)
    l.setLevel(logging.INFO)
    l.addFilter(PollingFilter())

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
if not TELEGRAM_BOT_TOKEN:
    logger.error("❌ TELEGRAM_BOT_TOKEN not found in .env! Exiting.")
    sys.exit(1)

# Global Activity State (Smart Idle Tracking)
class GlobalState:
    is_busy = False
    last_activity = time.time()
    _lock = threading.Lock()
    
    @classmethod
    def set_busy(cls, busy: bool):
        with cls._lock:
            cls.is_busy = busy
            cls.last_activity = time.time()
    
    @classmethod
    def get_idleness(cls):
        with cls._lock:
            if cls.is_busy: return 0
            return time.time() - cls.last_activity

# Locking Mechanisms
file_locks = {}
fl_lock = threading.Lock()

@contextmanager
def file_lock(path_str):
    """
    Simple in-process file/path locking.
    """
    path_str = str(path_str)
    with fl_lock:
        if path_str not in file_locks:
            file_locks[path_str] = threading.Lock()
        lock = file_locks[path_str]
    
    acquired = lock.acquire(timeout=LOCK_WAIT_SECS)
    try:
        if not acquired:
            logger.warning(f"🔒 Could not acquire lock for {path_str} in {LOCK_WAIT_SECS}s. Proceeding anyway (Split Brain Risk).")
        yield acquired
    finally:
        if acquired:
            lock.release()

def atomic_write(target_path, content, mode="w", encoding="utf-8"):
    """
    Atomic write using tempfile and os.replace.
    Includes robustness for Windows file locking (WinError 5/32).
    """
    target_path = Path(target_path)
    # Write to a temp file in the same directory (to ensure same filesystem for atomic rename)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    fd, temp_path = tempfile.mkstemp(dir=target_path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, mode, encoding=encoding) as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception: pass # Some systems/pipes don't support fsync
            
        # Atomic Rename with Retry
        max_retries = 3
        last_error = None
        
        for i in range(max_retries):
            try:
                os.replace(temp_path, target_path)
                return # Success
            except OSError as e:
                last_error = e
                # WinError 5: Access denied, WinError 32: Used by process
                # If these occur, we wait and try again or use fallback
                if getattr(e, 'winerror', 0) in [5, 32]:
                    time.sleep(0.5)
                    # Force delete strategy for Windows if standard replace fails
                    try:
                        if os.path.exists(target_path):
                            os.remove(target_path)
                        os.rename(temp_path, target_path)
                        return
                    except Exception:
                        pass # Retry standard loop
                elif i == max_retries - 1:
                    raise e
                    
        # If loop finishes without success
        if last_error: raise last_error
        
    except Exception as e:
        logger.error(f"❌ Atomic write failed: {e}")
        try:
            if os.path.exists(temp_path): os.remove(temp_path)
        except: pass




def sanitize_logs(text):
    """Redact sensitive keys from logs/debug artifacts."""
    if not isinstance(text, str): return text
    pattern = r'(?i)(token|key|secret|password|cookie|auth)\s*[:=]\s*["\']?([^"\',\s]+)["\']?'
    return re.sub(pattern, r'\1=***REDACTED***', text)

# Global State
user_sessions = {}
user_result_locks = {}
g_session_lock = threading.Lock()

def get_session_lock(user_id):
    with g_session_lock:
        if user_id not in user_result_locks:
            user_result_locks[user_id] = threading.Lock()
        return user_result_locks[user_id]

def save_session(user_id):
    """Persist individual session to disk."""
    if user_id in user_sessions:
        try:
            data = json.dumps(user_sessions[user_id], default=str)
            atomic_write(os.path.join(JOB_DIR, f"session_{user_id}.json"), data)
        except Exception as e:
            logger.error(f"Failed to persist session {user_id}: {e}")

def load_sessions():
    """Recover sessions from disk on startup."""
    try:
        now = time.time()
        count = 0
        for f in glob.glob(os.path.join(JOB_DIR, "session_*.json")):
            try:
                # Check age
                mtime = os.path.getmtime(f)
                if now - mtime > SESSION_TTL_SECS:
                    os.remove(f) # Expired
                    continue
                    
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    # Extract user_id from filename
                    fname = os.path.basename(f)
                    uid = int(fname.replace("session_", "").replace(".json", ""))
                    user_sessions[uid] = data
                    count += 1
            except Exception: pass
        logger.info(f"🔄 Restored {count} active sessions from disk.")
    except Exception as e:
        logger.warning(f"Session recovery failed: {e}")

# Global State
COMPILATION_BATCH_SIZE = int(os.getenv("COMPILATION_BATCH_SIZE", "5"))

# ==================== AUTO-INSTALL & SETUP ====================

# ==================== AUTO-INSTALL & SETUP ====================

# Cached Hardware Capabilites
_hardware_cache = None

def detect_hardware_capabilities():
    """
    Detect hardware capabilities (Cached) via ComputeCaps.
    """
    global _hardware_cache
    if _hardware_cache: return _hardware_cache
    
    from compute_caps import ComputeCaps
    caps = ComputeCaps.get()
    
    hardware_info = {
        'has_gpu': caps['has_cuda'] or caps['gpu_fast'], # Logical GPU presence
        'gpu_name': 'NVIDIA GPU' if caps['has_cuda'] else 'CPU',
        'vram_gb': caps['vram_gb'],
        'cuda_available': caps['has_cuda']
    }
    
    if hardware_info['has_gpu']:
         logger.info(f"🎮 GPU Detected via ComputeCaps: {hardware_info['gpu_name']} ({hardware_info['vram_gb']:.1f} GB VRAM)")
    else:
         logger.info("ℹ️ No GPU detected (ComputeCaps).")
         
    _hardware_cache = hardware_info
    return hardware_info

def resolve_compute_mode():
    """
    Resolve the final compute mode.
    Downgrades to CPU if VRAM is too low (< 6GB).
    """
    cpu_mode = os.getenv("CPU_MODE", "auto").lower()
    gpu_mode = os.getenv("GPU_MODE", "auto").lower()
    min_vram = int(os.getenv("MIN_VRAM_GB", "6"))
    
    # 1. Forced Modes
    if cpu_mode == "on":
        return "cpu"
    
    if gpu_mode == "on":
        return "gpu"
        
    # 2. Auto Logic
    hardware = detect_hardware_capabilities()
    
    if gpu_mode == "auto":
        if hardware['cuda_available']:
            # Check VRAM
            if hardware['vram_gb'] < min_vram:
                logger.info(f"⚠️ GPU detected but VRAM ({hardware['vram_gb']:.1f}GB) < {min_vram}GB. Falling back to CPU for stability.")
                return "cpu"
            
            logger.info(f"🤖 GPU_MODE=auto: CUDA ready & VRAM sufficient. Selecting GPU.")
            return "gpu"
            
    # Default fallback
    return "cpu"

def check_and_update_env():
    """
    Auto-updates .env file with missing keys and smart defaults.
    """
    env_path = ".env"
    if not os.path.exists(env_path):
        logger.warning("⚠️ .env file not found. Creating template...")
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("""# ==================== CORE SETTINGS ====================
# REQUIRED: Get your bot token from @BotFather on Telegram
TELEGRAM_BOT_TOKEN=YOUR_BOT_TOKEN_HERE

# REQUIRED: Get your API key from https://aistudio.google.com/app/apikey
GEMINI_API_KEY=YOUR_GEMINI_API_KEY_HERE

# ==================== PERFORMANCE ====================
# Modes: auto, on, off
CPU_MODE=auto
GPU_MODE=auto
REENCODE_PRESET=fast
REENCODE_CRF=25

# ==================== ENHANCEMENT ====================
ENHANCEMENT_LEVEL=medium
TARGET_RESOLUTION=1080:1920

# ==================== TRANSFORMATIVE FEATURES ====================
ADD_TEXT_OVERLAY=yes
TEXT_OVERLAY_TEXT=swargawasal
TEXT_OVERLAY_POSITION=bottom
TEXT_OVERLAY_STYLE=modern

ADD_COLOR_GRADING=yes
COLOR_FILTER=cinematic
COLOR_INTENSITY=0.5

ADD_SPEED_RAMPING=yes
SPEED_VARIATION=0.15

FORCE_AUDIO_REMIX=yes

# ==================== COMPILATION ====================
COMPILATION_BATCH_SIZE=6
SEND_TO_YOUTUBE=off
DEFAULT_HASHTAGS_SHORTS=#shorts #viral #trending
DEFAULT_HASHTAGS_COMPILATION=#compilation #funny #viral

# ==================== TRANSITIONS ====================
TRANSITION_DURATION=0.5
TRANSITION_INTERVAL=5
GEMINI_TITLE_COMPLICATION=on
""")
        logger.info("✅ Created .env template. Please update TELEGRAM_BOT_TOKEN and GEMINI_API_KEY!")
        
    # Load current env content
    with open(env_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    updates = []
    
    # Define required keys and defaults (HARDENED)
    required_keys = {
        "CPU_MODE": "auto",
        "GPU_MODE": "auto",
        "ENHANCEMENT_LEVEL": "medium",
        "TRANSITION_INTERVAL": "5",
        "TRANSITION_DURATION": "0.5",
        "FORCE_AUDIO_REMIX": "yes",
        "ADD_TEXT_OVERLAY": "yes",
        "ADD_SPEED_RAMPING": "yes",
        "NET_RETRY_COUNT": "3",
        "NET_BACKOFF_BASE": "2.0",
        "LOCK_WAIT_SECS": "5",
        "TELEGRAM_MAX_UPLOAD_MB": "50",
        "SESSION_TTL_SECS": "86400",
        "TELEGRAM_MAX_UPLOAD_MB": "50",
        "GEMINI_TITLE_COMPLICATION": "on",
        "ENABLE_COMMUNITY_POST_COMPILATION": "yes",
        "ENABLE_COMMUNITY_POST_SHORTS": "no",
    }
    
    for key, default in required_keys.items():
        if key not in os.environ and f"{key}=" not in content:
            logger.info(f"➕ Auto-adding missing key: {key}={default}")
            updates.append(f"\n# Auto-added by Smart Installer\n{key}={default}")
            os.environ[key] = default 
            
    if updates:
        with open(env_path, "a", encoding="utf-8") as f:
            f.writelines(updates)
        logger.info(f"✅ Auto-added {len(updates)} missing keys to .env")
        
    # Expose resolved compute mode
    cm = resolve_compute_mode()
    os.environ["COMPUTE_MODE"] = cm
    logger.info(f"🚀 FINAL COMPUTE MODE: {cm.upper()}")
    
    # 3. Heal JSON State Files
    check_and_heal_json_files()

def check_and_heal_json_files():
    """
    Auto-Heals missing JSON state/config files with intelligent defaults.
    Analyzes user behavior patterns to populate initial data where applicable.
    """
    
    # 1. cleanup_state.json
    # Tracks last cleanup time. Default: Never run checking.
    p_cleanup = "cleanup_state.json"
    if not os.path.exists(p_cleanup):
        try:
             with open(p_cleanup, 'w') as f:
                 json.dump({"last_run": 0}, f)
             logger.info(f"🩹 Auto-Healed: {p_cleanup}")
        except: pass

    # 2. community_promo_state.json
    # Tracks community post rate limits and hashes.
    p_promo = "community_promo_state.json"
    if not os.path.exists(p_promo):
        try:
             with open(p_promo, 'w') as f:
                 json.dump({"last_run": 0, "posted_hashes": []}, f)
             logger.info(f"🩹 Auto-Healed: {p_promo}")
        except: pass

    # 3. policy_memory.json
    # Tracks strategy success rates. Default: Empty memory.
    p_policy = "policy_memory.json"
    if not os.path.exists(p_policy):
        try:
             with open(p_policy, 'w') as f:
                 json.dump({}, f)
             logger.info(f"🩹 Auto-Healed: {p_policy}")
        except: pass

    # 4. caption_prompt.json
    # Stores the "Safe Fallback" caption.
    # We populate this with a high-quality "Transformative" example.
    p_caption = "caption_prompt.json"
    if not os.path.exists(p_caption):
        try:
             default_data = {
                 "caption_final": "Mixing vintage denim with modern confidence for a timeless look",
                 "last_source": "auto_healer",
                 "timestamp": datetime.now().isoformat()
             }
             with open(p_caption, 'w') as f:
                 json.dump(default_data, f, indent=2)
             logger.info(f"🩹 Auto-Healed: {p_caption}")
        except: pass

    # 5. title_expansion_presets.json
    # Presets for interactive title composition.
    # We populate this with "Viral/Clickbait" patterns tailored for Shorts.
    p_titles = "title_expansion_presets.json"
    if not os.path.exists(p_titles):
        try:
             presets = {
                 "1": { "label": "Wait for it... 😱", "suffix": " #waitforit" },
                 "2": { "label": "You won't believe this!", "suffix": " #shocking" },
                 "3": { "label": "Satisfying 😌", "suffix": " #satisfying" },
                 "4": { "label": "Viral Moment", "suffix": " #viral" },
                 "5": { "label": "Must Watch", "suffix": " #mustwatch" },
                 "6": { "label": "Relatable 😂", "suffix": " #relatable" }
             }
             with open(p_titles, 'w', encoding='utf-8') as f:
                 json.dump(presets, f, indent=2, ensure_ascii=False)
             logger.info(f"🩹 Auto-Healed: {p_titles}")
        except: pass

# Conditional imports removed for lazy loading
# compute_mode = os.environ.get("COMPUTE_MODE", "cpu") - moved to resolve_compute_mode if needed

# ==================== UTILS ====================

UPLOAD_LOG = "upload_log.csv"

def _ensure_log_header():
    if not os.path.exists(UPLOAD_LOG):
        with open(UPLOAD_LOG, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "video_id", "caption_style", "ypp_risk", "approved", "user_decision", "channel_name"])

def log_video(file_path: str, yt_link: str, title: str, style: str = "unknown", ypp_risk: str = "unknown", action: str = "approved", channel_name: str = "default_channel"):
    _ensure_log_header()
    # Atomic Append
    video_id = yt_link.split("/")[-1] if yt_link else "upload_failed"
    approved_bool = "true" if action == "approved" else "false"
    
    # Schema: timestamp, video_id, caption_style, ypp_risk, approved, user_decision, channel_name
    row = [datetime.utcnow().isoformat(), video_id, style, ypp_risk, approved_bool, action, channel_name]
    
    with file_lock(UPLOAD_LOG):
        with open(UPLOAD_LOG, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
            
    # Metadata JSON Sidecar
    try:
        final_meta = {
            "unique_id": video_id,
            "source_path": file_path,
            "youtube_link": yt_link,
            "title": title,
            "caption_style": style,
            "ypp_risk": ypp_risk,
            "user_decision": action,
            "channel_name": channel_name,
            "created_at": datetime.utcnow().isoformat(),
            "pipeline_version": "4.0-final-lock"
        }
        meta_path = str(file_path) + ".final.json"
        atomic_write(meta_path, json.dumps(final_meta))
    except Exception: pass

def total_uploads() -> int:
    if not os.path.exists(UPLOAD_LOG):
        return 0
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
        return max(0, len(rows) - 1)

def last_n_filepaths(n: int) -> list:
    """Get the last N video file paths from the upload log, filtered by recency."""
    if not os.path.exists(UPLOAD_LOG):
        return []
    
    with open(UPLOAD_LOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Filter by timestamp - only videos from last 24 hours
    from datetime import datetime, timedelta
    cutoff_time = datetime.utcnow() - timedelta(hours=24)
    
    recent_rows = []
    for r in rows:
        try:
            timestamp_str = r.get("timestamp", "")
            if timestamp_str:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if timestamp > cutoff_time:
                    recent_rows.append(r)
        except:
            # If timestamp parsing fails, skip this row
            continue
    
    # Get last N from recent rows
    subset = recent_rows[-n:]
    paths = [r.get("file_path") for r in subset if r.get("file_path")]
    
    # Return only paths that exist
    valid_paths = [p for p in paths if p and os.path.exists(p)]
    
    logger.info(f"📊 Found {len(valid_paths)} recent videos for compilation (last 24h)")
    return valid_paths

# Rate Limiting
class RateLimiter:
    def __init__(self, limit=10, period=60):
        self.limit = limit
        self.period = period
        self.users = {}
        self.lock = threading.Lock()
        
    def check(self, user_id):
        with self.lock:
            now = time.time()
            if user_id not in self.users:
                self.users[user_id] = []
            
            # Filter timestamps
            self.users[user_id] = [ts for ts in self.users[user_id] if now - ts < self.period]
            
            if len(self.users[user_id]) >= self.limit:
                return False
                
            self.users[user_id].append(now)
            return True

# Initialize Rate Limiter
user_limiter = RateLimiter(
    limit=int(os.getenv("USER_RATE_LIMIT_PER_MIN", "10")), 
    period=60
)

async def with_retry(func, *args, **kwargs):
    """
    Robust Retry Wrapper for Network Calls.
    """
    last_exception = None
    for attempt in range(NET_RETRY_COUNT):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            # Fail fast on 4xx (Client Error)
            msg = str(e)
            if "40" in msg or "400" in msg or "404" in msg or "403" in msg: 
                # Very rough heuristic, standard http libs usually provide status codes
                logger.error(f"❌ Non-Retriable Error: {e}")
                raise e
                
            wait = NET_BACKOFF_BASE ** attempt
            logger.warning(f"⚠️ Network Op Failed ({attempt+1}/{NET_RETRY_COUNT}): {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
            
    logger.error(f"❌ Network Op Failed after {NET_RETRY_COUNT} attempts.")
    raise last_exception

async def safe_reply(update: Update, text: str, force: bool = False):
    """
    Robust message sender with improved error handling and force-bypass for rate limits.
    Handles CallbackQuery updates gracefully.
    """
    try:
        user_id = update.effective_user.id
        
        # Rate Limit Check (Unless Forced)
        if not force and not user_limiter.check(user_id):
            logger.warning(f"🛑 Rate limit hit for user {user_id}")
            return

        for attempt in range(1, 4):
            try:
                # Handle CallbackQuery Logic (Where update.message might be None)
                target_msg = update.effective_message
                if not target_msg:
                    # Fallback for weird updates
                    if update.callback_query:
                         target_msg = update.callback_query.message
                
                if target_msg:
                    await target_msg.reply_text(
                        text,
                        read_timeout=30,
                        write_timeout=30,
                        connect_timeout=30,
                        pool_timeout=30
                    )
                else:
                    logger.warning("⚠️ safe_reply: No target message found to reply to.")
                    
                return
            except (NetworkError, TimedOut) as e:
                logger.warning(f"🛑 Reply failed (Attempt {attempt}/3): {e}. Retrying in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                # Catch BadRequest: "Message is not modified" or "Chat not found"
                # Do NOT retry fatal errors
                logger.warning(f"⚠️ safe_reply fatal error (No Retry): {e}")
                return
                
        logger.error("❌ Failed to send message after retries.")
        
    except Exception as e:
        logger.error(f"❌ safe_reply Crashed: {e}", exc_info=True)

class ProgressFile(io.BufferedReader):
    def __init__(self, filename, logger_func):
        self._size = os.path.getsize(filename)
        self._seen = 0
        self._last_log = 0
        self._logger = logger_func
        self._path = filename
        f = open(filename, 'rb')
        super().__init__(f)

    def read(self, size=-1):
        chunk = super().read(size)
        if chunk:
            self._seen += len(chunk)
            pct = (self._seen / self._size) * 100
            if pct - self._last_log >= 10:
                self._logger(f"📤 Telegram Upload: {pct:.0f}% ({os.path.basename(self._path)})")
                self._last_log = pct
        return chunk

async def safe_video_reply(update: Update, video_path: str, caption: str = None):
    """
    Robust video sender with retry logic and progress logging.
    """
    user_id = update.effective_user.id
    if not user_limiter.check(user_id): return

    try:
        f_size = os.path.getsize(video_path)
        if f_size == 0:
            logger.error(f"❌ Critical: Video file is 0 bytes! Cannot send. ({video_path})")
            await safe_reply(update, "❌ Processing Error: Resulting video is empty (0 bytes). Check logs.")
            return

        size_mb = f_size / (1024 * 1024)
        if size_mb > TELEGRAM_MAX_UPLOAD_MB:
             await safe_reply(update, f"⚠️ Video is {size_mb:.1f}MB (Max {TELEGRAM_MAX_UPLOAD_MB}MB). Link/File saved locally.")
             return
    except Exception as e:
        logger.error(f"Failed size check: {e}")
        pass

    for attempt in range(1, 4):
        try:
            if update.message:
                pf = ProgressFile(video_path, logger.info)
                await update.message.reply_video(
                    video=pf, 
                    caption=caption, 
                    read_timeout=600, 
                    write_timeout=600,
                    connect_timeout=60,
                    pool_timeout=60
                )
                pf.close()
            return
        except (NetworkError, TimedOut) as e:
            logger.warning(f"🛑 Video reply failed (Attempt {attempt}/3): {e}. Retrying in 5s...")
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"❌ Video reply error: {e}")
            break
            
    logger.error("❌ Failed to send video after retries.")
    await safe_reply(update, "❌ Failed to send video due to network timeout.")

def _validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        return any(allowed in domain for allowed in ALLOWED_DOMAINS)
    except: return False

def _sanitize_title(title: str) -> str:
    # Allow spaces but remove other special characters
    clean = re.sub(r'[^\w\s-]', '', title)
    # clean = clean.replace(' ', '_')  <-- REMOVED: Keep spaces for YouTube title
    return clean[:100]  # Increased limit slightly for better titles

def _get_hashtags(text: str) -> str:
    link_count = len(re.findall(r'https?://', text))
    if link_count > 1:
        return os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").strip()
    return os.getenv("DEFAULT_HASHTAGS_SHORTS", "").strip()



    return os.getenv("DEFAULT_HASHTAGS_SHORTS", "").strip()



# Helper for Incremental Filenaming
def _generate_next_filename(directory: str, prefix: str, extension: str = ".mp4") -> str:
    """
    Scans directory for files matching prefix_XX.mp4 and returns the next incremental filename.
    Format: prefix_01.mp4, prefix_02.mp4, etc.
    """
    try:
        if not os.path.exists(directory): return os.path.join(directory, f"{prefix}_01{extension}")
        
        # List all possible matches
        # We look for files starting with prefix
        candidates = glob.glob(os.path.join(directory, f"{prefix}_*{extension}"))
        
        max_idx = 0
        
        # Regex to extract the number at the end
        # We expect: prefix_(\d+).mp4
        # We must be careful not to match prefix_2025... as a huge number if the prefix matches partially.
        # So we ensure the prefix is followed by an UNDERSCORE and then DIGITS only.
        # But wait, our prefix might result in "compile_last_2" and we want "compile_last_2_01".
        # So pattern is: prefix + "_" + digits + extension
        
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+){re.escape(extension)}$")
        
        for f in candidates:
            fname = os.path.basename(f)
            match = pattern.match(fname)
            if match:
                try:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except: pass
                
        # If no strict match found (e.g. only timestamped files exist), we start at 01.
        # Timestamped files (prefix_2025...) won't match the regex `_(\d+).mp4` easily 
        # unless user named it `compile_last_2_20251228`. 
        # But timestamp usually has time too: `20251228_123456`. That contains `_`, so `\d+` won't match it fully if strict anchor.
        
        return os.path.join(directory, f"{prefix}_{max_idx+1:02d}{extension}")
        
    except Exception as e:
        logger.error(f"Filename generation error: {e}")
        # Fallback to timestamp if logic fails
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return os.path.join(directory, f"{prefix}_{stamp}{extension}")


async def initiate_compilation_title_flow(update: Update, merged_path: str, n_videos: int, hashtags: str, base_title: str = None):
    """
    New Flow: 
    1. Check GEMINI_TITLE_COMPLICATION
    2. ON -> Try Gemini -> Finish
    3. FAIL/OFF -> Ask User (Mandatory) -> Wait
    """
    user_id = update.effective_user.id
    gemini_mode = os.getenv("GEMINI_TITLE_COMPLICATION", "on").lower()
    
    generated_title = None
    
    # Defaults
    if not base_title:
        base_title = f"Compilation {n_videos} Videos"
    
    if gemini_mode == "on":
        try:
             # Try smart generation via Brain if valid base context
             from monetization_brain import brain
             # Construct context for brain from base_title if it looks like a query
             context = base_title.replace("Compilation", "").replace("Videos", "").strip()
             if not context: context = "Influencer Fashion"
             
             logger.info(f"🧠 Generating Compilation Title via Brain: {context}")
             smart = brain.generate_editorial_title(context)
             
             # Smart is now likely a tuple (title, desc) if updated
             if isinstance(smart, tuple):
                 title_cand, desc_cand = smart
             else:
                 title_cand, desc_cand = smart, None
                 
             if title_cand and title_cand != f"Compilation: {context}":
                 generated_title = title_cand
                 generated_desc = desc_cand
                 
        except Exception as e:
             logger.warning(f"Gemini Title Gen Failed: {e}")
    
    if generated_title:
        await safe_reply(update, f"✨ AI Generated Title: {generated_title}")
        await finish_compilation_upload(update, merged_path, generated_title, hashtags, n_videos=n_videos, description=generated_desc)
        return
        
    # --- FALLBACK: ASK USER (MANDATORY) ---
    presets_msg = ""
    try:
        if os.path.exists("title_expansion_presets.json"):
            with open("title_expansion_presets.json", "r", encoding="utf-8") as f:
                presets = json.load(f)
            
            if presets:
                msg_lines = [f"📌 Select title expansion for: '{base_title}' (optional):"]
                # Ensure sorted keys
                for k in sorted(presets.keys(), key=lambda x: int(x) if x.isdigit() else 99):
                    v = presets[k]
                    msg_lines.append(f"{k}️⃣ {v['label']}")
                msg_lines.append("\nReply with number or /skip")
                presets_msg = "\n".join(msg_lines)
    except Exception as e:
         logger.error(f"Failed to load presets: {e}")
         
    if presets_msg:
        # Save State
        with get_session_lock(user_id):
            user_sessions[user_id] = {
                'state': 'WAITING_FOR_COMPILATION_TITLE',
                'pending_compilation_path': merged_path,
                'pending_n_videos': n_videos,
                'pending_hashtags': hashtags,
                'pending_base_title': base_title 
            }
            save_session(user_id)
            
        await safe_reply(update, presets_msg)
    else:
        # No presets found? Fallback to generic
        await finish_compilation_upload(update, merged_path, base_title, hashtags, n_videos=n_videos)


async def finish_compilation_upload(update: Update, merged_path: str, title: str, hashtags: str, n_videos: int = 10, description: str = None):
    """
    Final step: Upload, Log, Reply.
    """
    # Explicitly log the final location for user clarity
    logger.info(f"💾 Compilation Saved Confirmation: {merged_path}")
    
    import uploader
    import community_promoter
    
    # Check if we should send to YouTube or Telegram
    try:
        send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() in ["on", "yes", "true"]
        
        link = None
        yt_status_msg = "🚫 YouTube: Skipped"

        if send_to_youtube:
            await safe_reply(update, f"📤 Uploading compilation: '{title}'...")
            
            try:
                # 1. YouTube Upload
                link = await with_retry(
                    uploader.upload_to_youtube,
                    merged_path, 
                    hashtags=hashtags, 
                    title=title,
                    description=description
                )

                if link:
                    log_video(merged_path, link, title)
                    yt_status_msg = f"✅ YouTube: Uploaded! ({link})"
                    
                    # Reset/Clear user session if strictly compilation (optional, but good hygiene)
                    user_id = update.effective_user.id
                    with get_session_lock(user_id):
                            # Only clear if we were in the waiting state
                            if user_sessions.get(user_id, {}).get('state') == 'WAITING_FOR_COMPILATION_TITLE':
                                user_sessions.pop(user_id, None)
                                save_session(user_id)
                else:
                    yt_status_msg = "❌ YouTube: Failed."
            except Exception as e:
                 logger.error(f"YouTube Upload Failed: {e}")
                 yt_status_msg = f"❌ YouTube Error: {e}"
        else:
             await safe_reply(update, f"✅ Compilation saved locally (YouTube Skipped):\n`{merged_path}`")

        # 2. Meta Upload (Instagram + Facebook)
        # Independent of YouTube failure (as per requirement)
        import meta_uploader
        meta_results = {}
        if os.getenv("ENABLE_META_UPLOAD", "no").lower() in ["yes", "true", "on"]:
                await safe_reply(update, "📤 Attempting Meta (Instagram/Facebook) Uploads...")
                # Use generated description or title for caption
                # For compilations, maybe use title + hashtags
                meta_caption = f"{title}\n\n{hashtags}"
                if description: meta_caption = f"{title}\n\n{description}\n\n{hashtags}"
                
                # --- FACEBOOK TITLE TRANSFORMATION ---
                fb_caption = meta_caption # Default fallback
                try:
                    # Load Mappings
                    fb_map_file = "title_expansion_fb.json"
                    presets_file = "title_expansion_presets.json"
                    
                    if os.path.exists(fb_map_file) and os.path.exists(presets_file):
                        with open(fb_map_file, "r", encoding="utf-8") as f: fb_presets = json.load(f)
                        with open(presets_file, "r", encoding="utf-8") as f: main_presets = json.load(f)
                        
                        # Find which preset was used in the title
                        found_key = None
                        for k, v in main_presets.items():
                            # Check if the Main Preset's Label is in the current title
                            # e.g. Title: "Disha Patani: Red Carpet Event" -> Label: "Red Carpet Event"
                            if v['label'] in title:
                                found_key = k
                                break
                        
                        if found_key and found_key in fb_presets:
                            # Map to FB Title
                            clean_fb_title = fb_presets[found_key]['label']
                            # Re-construct caption for FB: Clean Title + Hashtags (No description spam)
                            fb_caption = f"{clean_fb_title}\n\n{hashtags}"
                            logger.info(f"📘 Facebook Title Swapped: '{title}' -> '{clean_fb_title}'")
                except Exception as e:
                    logger.warning(f"FB Title Mapping Failed: {e}")

                meta_results = await meta_uploader.AsyncMetaUploader.upload_to_meta(
                    merged_path, 
                    meta_caption,
                    upload_type=os.getenv("META_UPLOAD_TYPE", "Reels"),
                    facebook_caption=fb_caption
                )
        
        # 3. Final Report
        report_lines = [f"🎉 Compilation Processing Complete!", ""]
        report_lines.append(yt_status_msg)
        
        if meta_results:
            # Instagram
            ig_res = meta_results.get("instagram", {"status": "skipped"})
            if isinstance(ig_res, str): ig_res = {"status": ig_res}
            ig_status = ig_res.get("status", "skipped")
            ig_link = ig_res.get("link", "")
            icon_ig = "✅" if ig_status == "success" else "❌" if "failed" in ig_status else "⏩"
            line_ig = f"{icon_ig} Instagram: {ig_status}"
            if ig_link: line_ig += f" ({ig_link})"
            report_lines.append(line_ig)
            
            # Facebook
            fb_res = meta_results.get("facebook", {"status": "skipped"})
            if isinstance(fb_res, str): fb_res = {"status": fb_res}
            fb_status = fb_res.get("status", "skipped")
            fb_link = fb_res.get("link", "")
            icon_fb = "✅" if fb_status == "success" else "❌" if "failed" in fb_status else "⏩"
            line_fb = f"{icon_fb} Facebook: {fb_status}"
            if fb_link: line_fb += f" ({fb_link})"
            report_lines.append(line_fb)
            
        await safe_reply(update, "\n".join(report_lines))

            
        # --- COMMUNITY PROMOTION ADD-ON ---
        if link and os.getenv("ENABLE_COMMUNITY_POST_COMPILATION", "yes").lower() == "yes":
            # Just REGISTER the link for future shorts. Do NOT post comment on the compilation itself.
            logger.info("💾 Registering Compilation Link for future cross-promotion...")
            community_promoter.promoter.register_compilation_url(link)
            
    except Exception as e:
        logger.exception("Upload failed: %s", e)
        await safe_reply(update, f"❌ Pipeline failed: {e}")


# ==================== COMPILATION LOGIC ====================

def last_n_filepaths(n=5):
    """
    Robustly find the last N processed videos for compilation.
    Checks 'Processed Shorts' directory.
    """
    source_dir = "Processed Shorts"
    if not os.path.exists(source_dir):
        logger.warning(f"last_n_filepaths: {source_dir} does not exist.")
        return []
        
    all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
    # Filter out compilations AND invalid 0-byte files
    valid_files = [f for f in all_files 
                   if "compile" not in os.path.basename(f) 
                   and "compilation" not in os.path.basename(f)
                   and os.path.getsize(f) > 1024]
    
    # Sort by modification time (Newest -> Oldest)
    valid_files.sort(key=os.path.getmtime, reverse=True)
    
    logger.info(f"📊 Found {len(valid_files)} recent videos for compilation")
    return valid_files[:n]


async def maybe_compile_and_upload(update: Update):
    from compiler import compile_batch_with_transitions
    import uploader
    count = total_uploads()
    n = COMPILATION_BATCH_SIZE
    if n <= 0 or count == 0 or count % n != 0:
        return

    await safe_reply(update, f"⏳ Creating compilation of last {n} shorts...📦")
    files = last_n_filepaths(n)
    if len(files) < n:
        await safe_reply(update, "⚠️ Not enough local files to compile. Skipping.")
        return

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_name = os.path.join(COMPILATIONS_DIR, f"compilation_{n}_{stamp}.mp4")
    await safe_reply(update, f"🔨 Merging {len(files)} videos now...🛸")

    try:
        await safe_reply(update, "✨ Running full AI pipeline for batch compilation…")

        # --- Single Stage: Batch Compile with Transitions ---
        # This replaces the old 2-stage process (raw merge -> enhance)
        # Now we normalize -> transition -> merge -> remix -> assemble in one go
        
        # Use Output Name directly (contains Path)
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            files,
            output_name
        )
        
        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Failed to create compilation.")
            return

        # Prepare Metadata
        count = total_uploads()
        # Default Title (will be overridden by logic likely, but passed as backup or logic param)
        # Actually logic generates title. We just need hashtags.
        
        comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "").replace("#Shorts", "").replace("#shorts", "").strip()
        
        # Initiate New Flow
        await initiate_compilation_title_flow(update, merged, n, comp_hashtags)

    except Exception as e:
        logger.exception("Compilation/upload failed: %s", e)
        await safe_reply(update, f"❌ Compilation failed: {e}")

async def compile_last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compiles the last N downloaded videos from the downloads/ folder.
    Usage: 
      /compile_last <number> (default 6)
      /compile_last <number> <name_prefix> (e.g. /compile_last 6 reem hot)
    """
    try:
        from compiler import compile_batch_with_transitions
        import uploader
        # 1. Parse arguments
        n = 6
        name_query = None
        
        if context.args:
            try:
                n = int(context.args[0])
            except ValueError:
                await safe_reply(update, "⚠️ Invalid number. Using default: 6")
            
            if len(context.args) > 1:
                name_query = " ".join(context.args[1:])
        
        if n <= 1:
            await safe_reply(update, "⚠️ Please specify at least 2 videos.")
            return

        # Source from Processed Shorts
        source_dir = "Processed Shorts"
        if not os.path.exists(source_dir):
             await safe_reply(update, f"❌ Directory '{source_dir}' not found.")
             return

        selected_files = []
        
        if name_query:
            # --- NAMED SORT COMPILATION ---
            # User wants specific named clips (e.g. reem_hot_1, reem_hot_2...)
            clean_query = _sanitize_title(name_query) # Use same sanitizer as downloader/main
            clean_query = clean_query.replace(' ', '_') # Ensure underscores if sanitizer kept spaces
            
            logger.info(f"🔍 Searching for clips matching: {clean_query}")
            await safe_reply(update, f"🔍 Searching for {n} clips matching '{clean_query}'...")
            
            # Find all files matching the pattern
            # We look for: base_name.mp4, base_name_1.mp4, base_name_2.mp4...
            # Or just any file starting with base_name
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            
            # Filter by name prefix
            matching_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if fname.startswith(clean_query):
                    matching_files.append(f)
            
            # Sort them naturally (reem_hot.mp4, reem_hot_1.mp4, reem_hot_2.mp4...)
            # We need smart sorting to handle _1, _2, _10 correctly
            def natural_keys(text):
                return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
                
            matching_files.sort(key=lambda f: natural_keys(os.path.basename(f)))
            
            if len(matching_files) < n:
                await safe_reply(update, f"⚠️ Not enough clips found matching '{clean_query}'. Found {len(matching_files)}, need {n}.")
                return
                
            # Take the first N (assuming user wants the sequence 1..N)
            # Or should we take the last N? 
            # User said: "reem_hot_1 + reem_hot_2 ... reem_hot_6"
            # This implies the first 6 of that sequence.
            # But if they downloaded 12, and ask for 6, maybe they want the latest?
            # "compile_last" usually means latest.
            # However, with named clips, usually you download a batch and want to compile THAT batch.
            # Let's take the LAST N to be consistent with command name.
            selected_files = matching_files[-n:]
            
        else:
            # --- DEFAULT: TIME BASED ---
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            files = [f for f in all_files if not os.path.basename(f).startswith("compile_")]
            
            if not files:
                await safe_reply(update, f"❌ No processed videos found in '{source_dir}' folder.")
                return
    
            # Sort by modification time (newest first)
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Take top N
            selected_files = files[:n]
        
        if len(selected_files) < 2:
            await safe_reply(update, f"⚠️ Found {len(selected_files)} videos, but need at least 2 to compile.")
            return

        # Log selected files for user confirmation
        msg = f"✅ Found {len(selected_files)} videos:\n"
        for f in selected_files:
            msg += f"- {os.path.basename(f)}\n"
        await safe_reply(update, msg)

        # 4. Compile
        if name_query:
            prefix = f"compile_last_{n}_{clean_query}"
        else:
            prefix = f"compile_last_{n}"
            
        output_filename = _generate_next_filename(COMPILATIONS_DIR, prefix, ".mp4")
        
        await safe_reply(update, "🚀 Starting batch compilation with transitions...")
        GlobalState.set_busy(True)
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            selected_files,
            output_filename
        )
        GlobalState.set_busy(False)

        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Compilation failed (check logs).")
            return

        # Prepare Hashtags
        comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "#compilation #viral").replace("#Shorts", "").strip()

        # If user provided a name query, use smart logic
        if name_query:
            # Smart Title Generation via Brain
            # Logic: Try Brain -> If Fail -> Initiate Title Flow (Fallback)
            
            try:
                from monetization_brain import brain
                logger.info(f"🧠 Generating Smart Title for: {name_query}")
                smart_res = brain.generate_editorial_title(name_query)
                
                # Unpack tuple
                if isinstance(smart_res, tuple):
                    smart_title, smart_desc = smart_res
                else:
                    smart_title, smart_desc = smart_res, None
                
                # Check for Failure
                is_fallback = (smart_title == f"Compilation: {name_query}")
                
                if smart_title and not is_fallback and len(smart_title) > 5:
                    final_title = smart_title
                    await finish_compilation_upload(update, merged, final_title, comp_hashtags, description=smart_desc)
                else:
                    # Smart Gen Failed -> Ask User
                    await initiate_compilation_title_flow(update, merged, len(selected_files), comp_hashtags, base_title=name_query)
                    
            except Exception as e:
                logger.warning(f"⚠️ Smart Title Generation Failed: {e}")
                await initiate_compilation_title_flow(update, merged, len(selected_files), comp_hashtags, base_title=name_query)
                
        else:
            # New Flow (No base name provided)
            await initiate_compilation_title_flow(update, merged, len(selected_files), comp_hashtags)

    except Exception as e:
        logger.exception(f"/compile_last failed: {e}")
        await safe_reply(update, f"❌ Error: {e}")

async def register_promo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Manually register a compilation URL for cross-promotion.
    Usage: /register_promo <url>
    """
    try:
        if not context.args:
            await safe_reply(update, "⚠️ Usage: /register_promo <youtube_url>")
            return
            
        url = context.args[0]
        import community_promoter
        community_promoter.promoter.register_compilation_url(url)
        await safe_reply(update, f"✅ Promotion Link Registered!\nTarget: {url}\nFuture Shorts will link to this.")
        
    except Exception as e:
        logger.error(f"Register Promo Failed: {e}")
        await safe_reply(update, f"❌ Error: {e}")

async def compile_first(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Compiles the FIRST N downloaded videos from the downloads/ folder.
    Usage: 
      /compile_first <number> (default 6)
      /compile_first <number> <name_prefix> (e.g. /compile_first 6 reem hot)
    """
    try:
        from compiler import compile_batch_with_transitions
        import uploader
        # 1. Parse arguments
        n = 6
        name_query = None
        
        if context.args:
            try:
                n = int(context.args[0])
            except ValueError:
                await safe_reply(update, "⚠️ Invalid number. Using default: 6")
            
            if len(context.args) > 1:
                name_query = " ".join(context.args[1:])
        
        if n <= 1:
            await safe_reply(update, "⚠️ Please specify at least 2 videos.")
            return

        # Source from Processed Shorts
        source_dir = "Processed Shorts"
        if not os.path.exists(source_dir):
             await safe_reply(update, f"❌ Directory '{source_dir}' not found.")
             return

        selected_files = []
        
        if name_query:
            # --- NAMED SORT COMPILATION ---
            clean_query = _sanitize_title(name_query)
            clean_query = clean_query.replace(' ', '_')
            
            logger.info(f"🔍 Searching for clips matching: {clean_query}")
            await safe_reply(update, f"🔍 Searching for {n} clips matching '{clean_query}'...")
            
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            
            # Filter by name prefix
            matching_files = []
            for f in all_files:
                fname = os.path.basename(f)
                if fname.startswith(clean_query):
                    matching_files.append(f)
            
            # Sort them naturally (1, 2, 3...)
            def natural_keys(text):
                return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]
                
            matching_files.sort(key=lambda f: natural_keys(os.path.basename(f)))
            
            if len(matching_files) < n:
                await safe_reply(update, f"⚠️ Not enough clips found matching '{clean_query}'. Found {len(matching_files)}, need {n}.")
                return
                
            # Take the FIRST N (1..N)
            selected_files = matching_files[:n]
            
        else:
            # --- DEFAULT: TIME BASED ---
            all_files = glob.glob(os.path.join(source_dir, "*.mp4"))
            files = [f for f in all_files if not os.path.basename(f).startswith("compile_")]
            
            if not files:
                await safe_reply(update, f"❌ No processed videos found in '{source_dir}' folder.")
                return
    
            # Sort by modification time (OLDEST first)
            files.sort(key=os.path.getmtime, reverse=False)
            
            # Take top N (which are now the oldest)
            selected_files = files[:n]
        
        if len(selected_files) < 2:
            await safe_reply(update, f"⚠️ Found {len(selected_files)} videos, but need at least 2 to compile.")
            return

        # Log selected files for user confirmation
        msg = f"✅ Found {len(selected_files)} videos:\n"
        for f in selected_files:
            msg += f"- {os.path.basename(f)}\n"
        await safe_reply(update, msg)

        # 4. Compile
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(COMPILATIONS_DIR, f"compile_first_{n}_{stamp}.mp4")
        if name_query:
            output_filename = os.path.join(COMPILATIONS_DIR, f"compile_{clean_query}_first_{n}_{stamp}.mp4")
        
        await safe_reply(update, "🚀 Starting batch compilation with transitions...")
        GlobalState.set_busy(True)
        merged = await asyncio.to_thread(
            compile_batch_with_transitions,
            selected_files,
            output_filename
        )
        GlobalState.set_busy(False)

        if not merged or not os.path.exists(merged):
            await safe_reply(update, "❌ Compilation failed (check logs).")
            return

        # Prepare Hashtags
        comp_hashtags = os.getenv("DEFAULT_HASHTAGS_COMPILATION", "#compilation #viral").replace("#Shorts", "").strip()

        if name_query:
            # Smart Title Generation via Brain
            # Logic: Try Brain -> If Fail -> Initiate Title Flow (Fallback)
            
            try:
                from monetization_brain import brain
                logger.info(f"🧠 Generating Smart Title for: {name_query}")
                smart_res = brain.generate_editorial_title(name_query)
                
                # Unpack tuple
                if isinstance(smart_res, tuple):
                    smart_title, smart_desc = smart_res
                else:
                    smart_title, smart_desc = smart_res, None
                
                # Check for Failure
                is_fallback = (smart_title == f"Compilation: {name_query}")
                
                if smart_title and not is_fallback and len(smart_title) > 5:
                    final_title = smart_title
                    await finish_compilation_upload(update, merged, final_title, comp_hashtags, description=smart_desc)
                else:
                    # Fail -> Fallback to User
                     await initiate_compilation_title_flow(update, merged, len(selected_files), comp_hashtags, base_title=name_query)
                     
            except Exception as e:
                logger.warning(f"⚠️ Smart Title Generation Failed: {e}")
                await initiate_compilation_title_flow(update, merged, len(selected_files), comp_hashtags, base_title=name_query)
                
        else:
            # New Flow
            await initiate_compilation_title_flow(update, merged, len(selected_files), comp_hashtags)

    except Exception as e:
        logger.exception(f"/compile_first failed: {e}")
        await safe_reply(update, f"❌ Error: {e}")

# ==================== HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, "❓ Please send an Instagram reel or YouTube link to begin.")

async def getbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await safe_reply(update, f"Current compilation batch size: {COMPILATION_BATCH_SIZE}")

async def setbatch(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global COMPILATION_BATCH_SIZE
    try:
        if not context.args:
            await safe_reply(update, "Usage: /setbatch <number>")
            return
        n = int(context.args[0])
        if n <= 0:
            await safe_reply(update, "Please provide a positive integer.")
            return
        COMPILATION_BATCH_SIZE = n
        await safe_reply(update, f"✅ Compilation batch size set to {n}.")
    except Exception:
        await safe_reply(update, "Usage: /setbatch <number>")

async def handle_attachment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles direct video file uploads (Video or Document).
    """
    print(f"DEBUG: Handle attachment triggered! Message ID: {update.message.message_id}")
    import compiler
    load_dotenv(override=True)
    
    user_id = update.effective_user.id
    message = update.message
    
    # Identify attachment
    attachment = message.video or message.document
    if not attachment:
        return # Should be filtered out by handlers but safe check
        
    # Filter non-video documents if needed
    if message.document:
        mime = getattr(attachment, 'mime_type', '')
        if not mime or not mime.startswith('video/'):
            await safe_reply(update, "⚠️ Document is not a recognized video format.")
            return

    file_name = getattr(attachment, 'file_name', f"upload_{int(time.time())}.mp4")
    
    # Check size (Telegram Bot API limit is 20MB for download, Local API is unlimited, MTProto is 2GB)
    # Using get_file() adheres to 20MB limit unless local API server used.
    # We'll try and catch error.
    file_size = getattr(attachment, 'file_size', 0)
    limit_mb = int(os.getenv("TELEGRAM_MAX_UPLOAD_MB", "50"))
    if file_size > limit_mb * 1024 * 1024:
         await safe_reply(update, f"⚠️ File is too large ({file_size/1024/1024:.1f}MB). Max: {limit_mb}MB.")
         return

    await safe_reply(update, "📥 Receiving video file...")
    
    try:
        new_file = await attachment.get_file()
        print(f"DEBUG: [Step 1] File object retrieved: {new_file.file_id}")
        
        # Sanitize filename
        clean_name = _sanitize_title(file_name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join("downloads", f"{clean_name}_{timestamp}.mp4")
        print(f"DEBUG: [Step 2] Save path generated: {save_path}")
        
        # Download
        print("DEBUG: [Step 3] Starting download...")
        await new_file.download_to_drive(save_path)
        print("DEBUG: [Step 4] Download completed!")
        
        # Setup Session for Title Input (Unified Flow)
        logger.info(f"💾 Setting up session for User {user_id} -> WAITING_FOR_TITLE")
        with get_session_lock(user_id):
             print(f"DEBUG: [Step 5] Session lock acquired for {user_id}")
             user_sessions[user_id] = {
                 'state': 'WAITING_FOR_TITLE',
                 'pending_local_path': str(save_path),
                 'pending_url': None # Explicitly clear URL
             }
             save_session(user_id)
             print("DEBUG: [Step 6] Session saved")
        
        # Ask for Title
        default_hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts")
        logger.info(f"📤 Sending Title Prompt to User {user_id}")
        await safe_reply(update, f"✅ File Received!\n\n📌 Hashtags:\n{default_hashtags}\n\n✏️ Now send the title to start processing.", force=True)
        print("DEBUG: [Step 7] Reply sent!")

    except Exception as e:
        logger.error(f"Attachment handler failed: {e}")
        
        # Smart Error Handling for Large Files
        if "File is too big" in str(e):
             await safe_reply(update, 
                 "⚠️ **Telegram API Limit Reached (20MB)**\n"
                 "Since I am running locally alongside your files, simply **Reply with the File Path** instead!\n\n"
                 "Example:\n"
                 "`D:\\Videos\\my_clip.mp4`"
             )
        else:
             await safe_reply(update, f"❌ Error handling file: {e}")
             
        GlobalState.set_busy(False)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import downloader
    import compiler
    load_dotenv(override=True)
    send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "off").lower() in ["on", "yes", "true"]
    
    text = update.message.text.strip()
    user_id = update.effective_user.id
    with get_session_lock(user_id):
        session = user_sessions.get(user_id, {})
        state = session.get('state')

    # Case 1: New URL
    if _validate_url(text):
        # Store URL and wait for title
        with get_session_lock(user_id):
            user_sessions[user_id] = {
                'state': 'WAITING_FOR_TITLE',
                'pending_url': text
            }
            save_session(user_id)
        
        default_hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts")
        
        await safe_reply(update, f"✅ Got the link!\n\n📌 Hashtags:\n{default_hashtags}\n\n✏️ Now send the title.")
        return

    # Case 1.5: Local File Path (Large File Bypass)
    # Check if text is a valid absolute path or relative path on the server
    possible_path = text.strip('"').strip("'") # Remove quotes if user added them
    if os.path.exists(possible_path) and os.path.isfile(possible_path):
         # Valid local file!
         file_name = os.path.basename(possible_path)
         file_size = os.path.getsize(possible_path)
         
         await safe_reply(update, f"📂 Found Local File: `{file_name}` ({file_size/1024/1024:.1f}MB)")
         
         # Copy to downloads to ensure isolation
         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
         clean_name = _sanitize_title(file_name)
         save_path = os.path.join("downloads", f"local_{clean_name}_{timestamp}.mp4")
         try:
             shutil.copy2(possible_path, save_path)
         except Exception as e:
             await safe_reply(update, f"❌ Failed to copy local file: {e}")
             return

         # Unified Flow: Go to Waiting For Title
         with get_session_lock(user_id):
             user_sessions[user_id] = {
                 'state': 'WAITING_FOR_TITLE',
                 'pending_local_path': str(save_path),
                 'pending_url': None
             }
             save_session(user_id)
             
         default_hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts")
         await safe_reply(update, f"✅ File Staged!\n\n📌 Hashtags:\n{default_hashtags}\n\n✏️ Now send the title to start processing.")
         return

    # Case 2: Waiting for Title
    if state == 'WAITING_FOR_TITLE':
        pending_url = session.get('pending_url')
        pending_local_path = session.get('pending_local_path')
        
        if not pending_url and not pending_local_path:
            await safe_reply(update, "❌ Error: No pending upload found. Please start over.")
            return
            
        custom_title = text
        await safe_reply(update, f"✅ Title set: '{custom_title}'\n✨ Processing...")
        
        video_path = None
        unique_filename = None
        url_hash = "local_upload"
        
        import hashlib
        
        # --- PATH A: PRE-DOWNLOADED FILE (Direct Upload) ---
        if pending_local_path:
             if os.path.exists(pending_local_path):
                 video_path = pending_local_path
                 # Generate pseudo-hash for consistency
                 url_hash = hashlib.md5(f"{pending_local_path}_{time.time()}".encode()).hexdigest()[:8]
                 # Rename to include Title for clarity? (Optional, but good for debugging)
                 # We'll stick to the existing path to avoid file errors.
             else:
                 await safe_reply(update, "❌ Error: Uploaded file verification failed. Please try again.")
                 return

        # --- PATH B: URL DOWNLOAD ---
        elif pending_url:
             await safe_reply(update, "📥 Downloading content...")
             
             # Generate Unique ID from URL
             url_hash = hashlib.md5(pending_url.encode()).hexdigest()[:8]
             
             # Sanitize title for filename
             clean_title = "".join([c for c in custom_title if c.isalnum() or c in (' ', '-', '_')]).strip()[:30]
             unique_filename = f"{clean_title}_{url_hash}.mp4"
             
             GlobalState.set_busy(True)
             
             # HARDENING: Strict Abort - No Retry
             video_path = await asyncio.to_thread(
                downloader.download_video, 
                pending_url, 
                custom_title=custom_title,
                force_filename=unique_filename
             )
             
             if not video_path:
                GlobalState.set_busy(False)
                await safe_reply(update, "❌ Download failed (Strict Abort).")
                with get_session_lock(user_id):
                    user_sessions.pop(user_id, None)
                    try: os.remove(os.path.join(JOB_DIR, f"session_{user_id}.json"))
                    except: pass
                return
        
        # --- COMMON PROCESSING ---
        if not video_path: 
             await safe_reply(update, "❌ Critical Error: Video path missing.")
             return

        # DEDUPLICATION CHECK (STEP 2)
        from deduplication import DedupEngine
        
        # Check for collision
        col_type, col_msg = DedupEngine.check_collision(url_hash, video_path)
        
        if col_type != "NONE":
            logger.warning(col_msg)
            logger.warning("⚠️ Content Collision Detected: Forcing FRESH processing pipeline.")
            meta_path = str(video_path) + ".json"
            if os.path.exists(meta_path):
                 try: os.remove(meta_path)
                 except: pass
        
        DedupEngine.register_content(url_hash, video_path, source="user_submission")

        # Load metadata (for hashtags etc)
        metadata = {}
        try:
            meta_path = os.path.splitext(video_path)[0] + ".json"
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            
        # Use user title, but sanitize it for display/files
        title = custom_title
        
        # Combine Metadata Tags + Default Hashtags
        meta_tags = metadata.get('tags', [])
        default_hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts #viral #trending")
        
        if meta_tags:
             # Take top 5 meta tags
             meta_tag_str = " ".join([f"#{t}" for t in meta_tags[:5]])
             hashtags = f"{default_hashtags} {meta_tag_str}"
        else:
             hashtags = default_hashtags
        
        # Store Downloaded Path for Retries (CRITICAL FOR NUCLEAR RETRY)
        with get_session_lock(user_id):
             if user_id not in user_sessions: user_sessions[user_id] = {}
             user_sessions[user_id]['source_path'] = str(video_path)
             # Bug fix: Ensure retry_count is initialized
             user_sessions[user_id]['retry_count'] = 0
             # Explicitly save title here too just in case
             user_sessions[user_id]['title'] = custom_title
             save_session(user_id)

        # Removed redundant "Downloaded" message here as we sent custom ones above
        
        # Compile/Process
        # Ensure we set busy if it wasn't set (Local Path case)
        GlobalState.set_busy(True)
        
        should_force = (col_type != "NONE")
        final_path, wm_context = await asyncio.to_thread(
             compiler.compile_with_transitions, 
             Path(video_path), 
             title, 
             force_reprocess=should_force
        )
        # --- OUTPUT STATE RESOLVER ---
        if not final_path or not os.path.exists(final_path):
             await safe_reply(update, "❌ Compilation failed (Critical Error).")
             # Clean session
             with get_session_lock(user_id):
                 user_sessions.pop(user_id, None)
             return

        final_str = str(final_path)

        # Enforce defaults if variables are somehow empty/None
        if not locals().get("wm_context"): wm_context = {}
        
        # Retrieve Sidecar Metadata (Ferrari Audit Fix)
        mon_meta = {}
        pipeline_metrics = {}
        opt_caption = None
        try:
             sidecar_path = os.path.splitext(final_str)[0] + ".json"
             if os.path.exists(sidecar_path):
                 with open(sidecar_path, 'r') as f:
                     sc_data = json.load(f)
                     pipeline_metrics = sc_data.get('pipeline_metrics', {})
                     mon_meta = pipeline_metrics.get('monetization', {})
                     if 'caption_data' in sc_data:
                         opt_caption = sc_data['caption_data'].get('caption')
        except: pass

        # Default Safety Values
        ypp_risk = mon_meta.get('risk_level', 'UNKNOWN')
        is_approved = (ypp_risk in ['LOW', 'MEDIUM'])
        style = "Transformative" # Default
        action = "APPROVE" if is_approved else "REVIEW"
        # Reason Safety (Check both Brain 'risk_reason' and Compiler 'reason')
        reason = mon_meta.get('risk_reason') or mon_meta.get('reason', 'Analysis pending or not performed.')

        # Watermark Status Derivation
        wm_status = wm_context.get('watermark_status', 'NOT_DETECTED')
        
        # Monetization Status Derivation
        monetization_status = "PASSED" if is_approved else "REVIEW"
        if ypp_risk == "HIGH": monetization_status = "BLOCKED"
        
        # Reason Safety (Fallback)
        if not reason: reason = "Transformative edit approved."

        # Dynamic Watermark Message (FINAL UPDATE)
        # Ensure message matches reality of wm_context even if exceptions occurred above
        wm_msg = "(No watermark detected - reply 'no' if missed)"
        final_status = wm_context.get('watermark_status')
        
        if final_status == "DETECTED_AND_REMOVED":
             wm_msg = "(Watermark detected & removed - verify result)"
        elif final_status == "DETECTED_BUT_SKIPPED":
             wm_msg = "(Watermark detected but skipped for safety)"
        elif final_status == "DETECTED_BUT_FAILED":
             wm_msg = "(Watermark removal FAILED - verify result)"

        # Caption Safety
        display_caption = opt_caption if opt_caption else title
        overlay_text = os.getenv('TEXT_OVERLAY_CONTENT', 'swargawasal') # Default from envy

        # Update Session with Brain Data
        with get_session_lock(user_id):
            user_sessions[user_id]['monetization_report'] = {
                "risk": ypp_risk,
                "style": style,
                "approved": is_approved,
                "action": action
            }
            # BUG FIX: Save the Final Video Path to session so /approve can find it
            user_sessions[user_id]['final_path'] = final_str
            user_sessions[user_id]['title'] = title # Update title too if needed
            
            # BUG FIX: Explicitly set state to WAITING_FOR_APPROVAL so commands work
            user_sessions[user_id]['state'] = 'WAITING_FOR_APPROVAL'
            save_session(user_id)
        
        await safe_reply(update, "✅ Video processed! Sending preview...")
        
        # STRICT FINAL REVIEW SUMMARY TEMPLATE
        report = (
            "--------------------------------\n"
            "🎬 FINAL REVIEW SUMMARY\n"
            "--------------------------------\n\n"
            f"🎯 Title:\n{title}\n\n"
            f"📌 Caption Generated:\n\"{display_caption}\"\n\n"
            f"🖋️ Text Overlays:\n"
            f"• fixed: {overlay_text}\n\n"
            f"🧠 Watermark Status:\n{wm_status}\n\n"
            f"💰 Monetization Status:\n{monetization_status}\n\n"
            f"⚠️ Risk Level:\n{ypp_risk}\n\n"
            f"🎨 Transformation:\n{mon_meta.get('transformation_score', 'N/A')}% ({mon_meta.get('verdict', 'N/A')})\n\n"
            f"📎 Reason:\n{reason}\n\n"
        )
        
        tips = mon_meta.get('improvement_tips', [])
        if tips:
            report += "💡 Improvement Tips:\n"
            for t in tips:
                report += f"• {t}\n"
            report += "\n"

        # Policy Citation (New)
        citation = mon_meta.get('policy_citation')
        if citation:
             report += f"📜 Policy Matched: \"{citation}\"\n\n"
            
        report += (
            "Reply /approve to upload or /reject to discard.\n"
            f"{wm_msg}\n"
            "--------------------------------"
        )
        
        # Use Brain Caption + Report
        caption = (
            f"{report}\n"  # Full report now includes instructions
        )
            
        try:
            if os.path.getsize(final_str) < 50 * 1024 * 1024:
                await safe_video_reply(update, final_str, caption=caption)
            else:
                await safe_reply(update, f"⚠️ Video too large for Telegram preview.\n{report}\nReply /approve to upload blindly or /reject.")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            await safe_reply(update, "❌ Error occurred.")
        return
            
    # Case 3.5: WAITING_FOR_COMPILATION_TITLE (New Mandatory Flow)
    if state == 'WAITING_FOR_COMPILATION_TITLE':
        user_id = update.effective_user.id
        
        # Retrieve context
        with get_session_lock(user_id):
            session = user_sessions.get(user_id, {})
            merged_path = session.get('pending_compilation_path')
            n_videos    = session.get('pending_n_videos')
            hashtags    = session.get('pending_hashtags')
            base_title  = session.get('pending_base_title', "")
            
        if not merged_path or not os.path.exists(merged_path):
             await safe_reply(update, "❌ Compilation file lost. Please try again.")
             with get_session_lock(user_id):
                 user_sessions.pop(user_id, None)
                 save_session(user_id)
             return

        title_choice = text.strip()
        final_title = ""
        
        # SKIP LOGIC
        if title_choice.lower().startswith("/skip"):
            final_title = base_title if base_title else f"Compilation {n_videos} Videos"
            await safe_reply(update, f"⏩ Skipping preset. Using Base Title: {final_title}")
            await finish_compilation_upload(update, merged_path, final_title, hashtags)
            return
            
        # PRESET LOGIC
        try:
             with open("title_expansion_presets.json", "r", encoding="utf-8") as f:
                 presets = json.load(f)
             
             if title_choice in presets:
                 item = presets[title_choice]
                 suffix = item.get('suffix', '')
                 # Logic: Base Title + Suffix (e.g. "Name" + " | Tag | Tag")
                 if base_title and base_title != f"Compilation {n_videos} Videos":
                     final_title = f"{base_title}{suffix}"
                 else:
                     # If generic base, just use Suffix (stripped of separator) or Label fallback
                     final_title = f"{item['label']} {suffix}"
                 
                 logger.info(f"DEBUG: Final Title Set To: '{final_title}'")
             else:
                 await safe_reply(update, "⚠️ Invalid selection. Please reply with the number (e.g., '1') or /skip.")
                 return
        except Exception as e:
             logger.error(f"Preset load error: {e}")
             final_title = base_title or f"Compilation {n_videos} Videos"

        await safe_reply(update, f"✅ Selected Title: {final_title}")
        
        # Proceed to Finish
        await finish_compilation_upload(update, merged_path, final_title, hashtags)
        return

    # Case 3: Title Expansion Selection (OLD - KEPT FOR BACKWARD COMPAT IF NEEDED or REMOVE?)
    # The user said: "ask user for tittle that from title_expansion_presets.json... but not as optional."
    # The old flow was optional after approval.
    # I will KEEP the old flow for single videos if it exists, but the new flow is for compilations.
    # The old flow state is 'WAITING_FOR_TITLE_EXPANSION', new is 'WAITING_FOR_COMPILATION_TITLE'.
    
    if state == 'WAITING_FOR_TITLE_EXPANSION':
        if text.startswith('/skip'):
             await _perform_upload(update, context)
        elif text.isdigit():
             # Load presets
             try:
                 with open("title_expansion_presets.json", "r", encoding="utf-8") as f:
                     presets = json.load(f)
                 choice = presets.get(text)
                 if choice:
                     suffix = choice.get("suffix", "")
                     # Update title in session
                     with get_session_lock(user_id):
                         current_title = user_sessions[user_id].get('title', "")
                         user_sessions[user_id]['title'] = f"{current_title}{suffix}"
                         save_session(user_id)
                     await safe_reply(update, f"✅ Title Updated: {user_sessions[user_id]['title']}")
                     await _perform_upload(update, context)
                 else:
                     await safe_reply(update, "⚠️ Invalid selection. Reply number or /skip.")
             except Exception:
                 await _perform_upload(update, context)
        else:
             await safe_reply(update, "⚠️ Reply with a number to apply preset, or /skip.")
        return

    # Case 4: Approval
    if state == 'WAITING_FOR_APPROVAL':
        if text.lower() in ['approve', '/approve']:
            await approve_upload(update, context)
        elif text.lower() in ['yes', 'y']:
            await verify_watermark(update, context, is_positive=True)
        elif text.lower() in ['no', 'n']:
            await verify_watermark(update, context, is_positive=False)
        elif text.lower() in ['reject', '/reject']:
            await reject_upload(update, context)
        else:
            await safe_reply(update, "⚠️ Options:\n• 'yes'/'no' - Verify watermark removal (Training Data)\n• '/approve' - Upload to YouTube\n• '/reject' - Discard Video")
        return

async def approve_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Step 1 of Approval: Ask for Title Expansion.
    """
    user_id = update.effective_user.id
    
    with get_session_lock(user_id):
        session = user_sessions.get(user_id, {})
        if session.get('state') != 'WAITING_FOR_APPROVAL':
            await safe_reply(update, "⚠️ No video waiting for approval.")
            return

    # Load Presets
    presets_msg = ""
    try:
        if os.path.exists("title_expansion_presets.json"):
            with open("title_expansion_presets.json", "r", encoding="utf-8") as f:
                presets = json.load(f)
            
            if presets:
                msg_lines = ["📌 Select title expansion (optional):"]
                for k, v in presets.items():
                    msg_lines.append(f"{k}️⃣ {v['label']}")
                msg_lines.append("\nReply with number or /skip")
                presets_msg = "\n".join(msg_lines)
    except Exception: pass

    if presets_msg:
        with get_session_lock(user_id):
            user_sessions[user_id]['state'] = 'WAITING_FOR_TITLE_EXPANSION'
            save_session(user_id)
        # FORCE REPLY to ensure user sees the menu even if they spammed buttons
        await safe_reply(update, presets_msg, force=True)
    else:
        # No presets, direct upload
        await _perform_upload(update, context)

async def _perform_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import uploader
    user_id = update.effective_user.id
    
    with get_session_lock(user_id):
        session = user_sessions.get(user_id, {})
        final_path = session.get('final_path')
        title = session.get('title')
        # existing hashtags from session (if any)
        hashtags = session.get('hashtags') 
    
    if not final_path or not os.path.exists(final_path):
        await safe_reply(update, "❌ Video file found missing during upload phase.")
        return

    # --- AI HASHTAG GENERATION LOGIC ---
    # 1. Logic Gate: Check Env
    enable_hashtag_gen = os.getenv("HASHTAG_GEN", "no").lower() == "yes"
    
    generated_hashtags = None
    if enable_hashtag_gen:
        try:
            await safe_reply(update, "🧠 Generating AI Hashtags...")
            # Run in thread to avoid blocking
            import gemini_captions
            generated_hashtags = await asyncio.to_thread(
                gemini_captions.generate_hashtags_from_video, 
                final_path
            )
        except Exception as e:
            logger.warning(f"Failed to generate hashtags: {e}")

    # 2. Resolution Strategy
    if generated_hashtags:
        hashtags = generated_hashtags
        await safe_reply(update, f"🏷️ Used AI Hashtags: {hashtags}")
    else:
        # Fallback to session or default
        if not hashtags:
             hashtags = os.getenv("DEFAULT_HASHTAGS_SHORTS", "#shorts #viral #trending")
             if enable_hashtag_gen:
                 await safe_reply(update, "⚠️ AI Hashtags failed. Using defaults.")

    # 3. YouTube Upload (Conditional)
    try:
        send_to_youtube = os.getenv("SEND_TO_YOUTUBE", "on").lower() in ["on", "yes", "true"]
        link = None
        yt_msg = "" # Initialize here to avoid unbound error
        
        if send_to_youtube:
            await safe_reply(update, "📤 Uploading to YouTube...", force=True)
            logger.info(f"🚀 Calling uploader for: {final_path}")
            try:
                # HARDENING: Retry Network Call
                link = await with_retry(uploader.upload_to_youtube, final_path, title=title, hashtags=hashtags)
                
                if link:
                    yt_msg = f"✅ YouTube: Success ({link})"
                    
                    # Log with strict monetization data
                    mon_data = session.get('monetization_report', {})
                    log_video(final_path, link, title, 
                              ypp_risk=mon_data.get('risk', 'unknown'),
                              style=mon_data.get('source', 'unknown'), # Log Source as Style for visibility
                              action="approved") # User clicked approve
                    
                    # --- COMMUNITY PROMOTION (SHORTS ONLY) ---
                    # Post a comment on this Short pointing to the last Compilation
                    if os.getenv("ENABLE_COMMUNITY_POST_COMPILATION", "yes").lower() == "yes":
                         logger.info("🚀 Triggering Cross-Promotion on Short (Background Task)...")
                         asyncio.create_task(
                             community_promoter.promoter.promote_on_short_async(
                                 uploader.get_authenticated_service(),
                                 link
                             )
                         )
                else:
                    yt_msg = "❌ YouTube: Failed"

            except Exception as e:
                logger.error(f"YouTube Upload Failed: {e}")
                yt_msg = f"❌ YouTube Error: {e}"
        else:
            logger.info("🚫 SEND_TO_YOUTUBE is OFF. Skipping YouTube upload.")
            await safe_reply(update, "⏭️ YouTube Upload Skipped (Configured OFF).")
            yt_msg = "⏩ YouTube: Skipped"
            
        # 2. Meta Upload (Runs INDEPENDENTLY of YouTube success/failure/skip)
        import meta_uploader
        meta_results = {}
        if os.getenv("ENABLE_META_UPLOAD", "no").lower() in ["yes", "true", "on"]:
             await safe_reply(update, "📤 Attempting Meta (Instagram/Facebook) Uploads...")
             # Construct Caption
             meta_caption = f"{title}\n\n{hashtags}" 
             
             meta_results = await meta_uploader.AsyncMetaUploader.upload_to_meta(
                 final_path, 
                 meta_caption,
                 upload_type=os.getenv("META_UPLOAD_TYPE", "Reels"),
                 skip_facebook=True # 🛑 RESTRICT FB TO COMPILATIONS ONLY
             )
             
        # 3. Final Report
        report_lines = ["🚀 Upload Summary:", ""]
        report_lines.append(yt_msg)
        
        if meta_results:
            # Instagram
            ig_res = meta_results.get("instagram", {"status": "skipped"})
            if isinstance(ig_res, str): ig_res = {"status": ig_res}
            ig_status = ig_res.get("status", "skipped")
            ig_link = ig_res.get("link", "")
            icon_ig = "✅" if ig_status == "success" else "❌" if "failed" in ig_status else "⏩"
            line_ig = f"{icon_ig} Instagram: {ig_status}"
            if ig_link: line_ig += f" ({ig_link})"
            report_lines.append(line_ig)
            
            # Facebook
            fb_res = meta_results.get("facebook", {"status": "skipped"})
            if isinstance(fb_res, str): fb_res = {"status": fb_res}
            fb_status = fb_res.get("status", "skipped")
            fb_link = fb_res.get("link", "")
            icon_fb = "✅" if fb_status == "success" else "❌" if "failed" in fb_status else "⏩"
            line_fb = f"{icon_fb} Facebook: {fb_status}"
            if fb_link: line_fb += f" ({fb_link})"
            report_lines.append(line_fb)
            
        await safe_reply(update, "\n".join(report_lines))

        # Check for compilation trigger
        if link: # Only trigger compile if at least youtube worked? Or always?
             # Logic: Compilation usually builds from "Processed Shorts".
             # If upload failed, the file is still in Processed Shorts?
             # Yes. So we can trigger it.
             await maybe_compile_and_upload(update)
             
    except Exception as e:
        logger.error(f"Upload error: {e}")
        await safe_reply(update, f"❌ Upload error: {e}")
        
    # Clear session
    with get_session_lock(user_id):
        user_sessions.pop(user_id, None)
        try: os.remove(os.path.join(JOB_DIR, f"session_{user_id}.json"))
        except: pass

async def verify_watermark(update: Update, context: ContextTypes.DEFAULT_TYPE, is_positive: bool = None):
    query = update.callback_query
    
    # Check if called via button (query) or command (arg)
    if query:
        await query.answer()
        user_id = query.from_user.id
        # Use query data mapping if arg is None
        if is_positive is None:
            if query.data == "wm_clean": is_positive = True
            elif query.data == "wm_bad": is_positive = False
    else:
        # Called via text command
        user_id = update.effective_user.id
    
    if is_positive is None:
        # Should not happen if logic is correct, but safety
        return

    import hybrid_watermark # Ensure imported for learning
    
    # Helper for robust editing (Text vs Caption)
    async def smart_edit(text):
        if not query:
            await safe_reply(update, text)
            return
            
        try:
            if query.message.text:
                await query.edit_message_text(text)
            elif query.message.caption is not None: # It's a media message
                await query.edit_message_caption(caption=text)
            else:
                # Fallback for weird cases (stickers? types without caption?)
                await safe_reply(update, text)
        except Exception as e:
            logger.warning(f"⚠️ Smart Edit Failed: {e}")
            await safe_reply(update, text)

    with get_session_lock(user_id):
        session = user_sessions.get(user_id, {})
        # Fallback Logic: handle_message sets 'final_path', retry sets 'pending_video'
        video_path = session.get('pending_video') or session.get('final_path')
        if not video_path:
             msg = "❌ Session expired (Video path lost). Please upload again."
             await smart_edit(msg)
             return
        title = session.get('title', 'video')
        # Retry Tracker
        retry_count = session.get('retry_count', 0)
        
        if is_positive:
            # Positive Feedback
            hybrid_watermark.hybrid_detector.confirm_learning(session.get("wm_context",{}), is_positive=True)
            
            msg = f"✅ Watermark Verification Successful! Proceeding to next step..."
            await smart_edit(msg)
            
            # PROCEED TO APPROVAL FLOW
            try:
                # ensuring state is correct for approve_upload check
                session['state'] = 'WAITING_FOR_APPROVAL'
                save_session(user_id)
                
                await approve_upload(update, context)
            except Exception as e:
                logger.error(f"❌ Error in Approval Flow trigger: {e}", exc_info=True)
                await safe_reply(update, "❌ Error proceeding to upload. Please try /approve manually.", force=True)
            return


        else:
            # Negative Feedback -> RETRY LOOP
            
            # 1. STRICT DELETION (Soft Reset)
            # We must delete the FAILED artifact to prevent pollution.
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    logger.info(f"🗑️ Strict Deletion (Rejected): {video_path}")
                
                # Try to delete associated JSON
                json_path = os.path.splitext(video_path)[0] + ".json"
                if os.path.exists(json_path):
                     os.remove(json_path)
                     logger.info(f"🗑️ Strict Deletion (Meta): {json_path}")
            except Exception as e:
                logger.warning(f"Deletion warning: {e}")

            # 2. Learning
            hybrid_watermark.hybrid_detector.confirm_learning(session.get("wm_context",{}), is_positive=False)
            
            # 3. Increment Level
            retry_count += 1
            session['retry_count'] = retry_count
            save_session(user_id)
            
            if retry_count > 2:
                # Max Retries Reached -> Give Up
                msg = "❌ Maximum retries reached. I'm sorry I couldn't clean it."
                await smart_edit(msg)
                user_sessions.pop(user_id, None)
                GlobalState.set_busy(False)
                return

            # 4. Trigger Retry
            # Level 1: Aggressive Static
            # Level 2: Better Accurate Patch (Static+6) OR Dynamic (if moving)
            
            mode_name = "AGGRESSIVE" if retry_count == 1 else "NUCLEAR_ENHANCED"
            status_msg = f"🔄 Retry {retry_count}/2: Activating {mode_name} Correction...\n(This might take longer)"
            await smart_edit(status_msg)
            
            # 5. Re-run Compiler
            # We assume pending_video WAS the input or we still have access to original download?
            # Actually, main.py usually keeps 'pending_url' or original download until finished.
            # But compiler overwrites? No, it makes a NEW file.
            # We need the path to the SOURCE video (downloaded raw).
            # Session usually has 'video_path' populated from download, and 'pending_video' populated from compile?
            # Let's use 'video_path' (downloaded) if available, else 'pending_video' (compiled) would be circular if deleted.
            
            # Wait, `handle_message` download stores path in `video_path` variable, but in SESSION?
            # We need to ensure we have the source.
            # Let's optimistically assume `session['source_path']` exists (I will add it in handle_message next step).
            # Fallback: If not, we might fail.
            
            source_path = session.get('source_path')
            
            # If source path is missing, we try to guess from session state or fail
            if not source_path:
                 await query.edit_message_text("❌ Error: Original source lost. Cannot retry.")
                 return
            
            try:
                import compiler
                retry_out, ctx = await asyncio.to_thread(
                    compiler.compile_with_transitions, 
                    Path(source_path), 
                    title, 
                    retry_level=retry_count
                )
                
                if retry_out:
                    # Update Session
                    session['pending_video'] = str(retry_out)
                    session['wm_context'] = ctx
                    save_session(user_id)
                    
                    # Ask Again
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    keyboard = [
                        [InlineKeyboardButton("✅ Perfect (Post It)", callback_data="wm_clean")],
                        [InlineKeyboardButton("❌ Still Bad (Retry)", callback_data="wm_bad")]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    
                    await context.bot.send_video(
                        chat_id=user_id,
                        video=open(retry_out, 'rb'),
                        caption=f"📝 Retry {retry_count} Result ({mode_name}).\nIs the watermark gone?",
                        reply_markup=reply_markup,
                        read_timeout=120, 
                        write_timeout=120,
                        connect_timeout=60
                    )
                else:
                    await query.edit_message_text("❌ Retry failed to produce output.")
                    
            except Exception as e:
                logger.error(f"Retry Error: {e}")
                await query.edit_message_text("❌ Error during retry.")

async def reject_upload(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("DEBUG: Entered reject_upload")
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"DEBUG: Entered reject_upload at {datetime.now()}\n")
        
    user_id = update.effective_user.id
    session = user_sessions.get(user_id, {})
    
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(f"DEBUG: Session state: {session.get('state')}\n")
    
    if session.get('state') == 'WAITING_FOR_APPROVAL':
        final_path = session.get('final_path')
        
        # User REJECTED: Permanent Delete
        if final_path and os.path.exists(final_path):
             try:
                 os.remove(final_path)
                 logger.info(f"🗑️ Deleted rejected file: {final_path}")
             except Exception as e:
                 logger.error(f"Failed to delete file: {e}")
             
             # Also delete sibling JSON if exists
             json_sibling = os.path.splitext(final_path)[0] + ".json"
             if os.path.exists(json_sibling):
                  try: os.remove(json_sibling)
                  except: pass
                  
             await safe_reply(update, "🗑️ Video permanently deleted.")
        else:
             await safe_reply(update, "🗑️ Video discarded (File missing).")
            
        print("DEBUG: Clearing session after reject")
        session_lock = get_session_lock(user_id)
        with session_lock:
            user_sessions.pop(user_id, None)
            # Remove persistence file
            try:
                os.remove(os.path.join(JOB_DIR, f"session_{user_id}.json"))
            except: pass
    else:
        print("DEBUG: Nothing to reject")
        with open("debug_log.txt", "a", encoding="utf-8") as f:
            f.write(f"DEBUG: Nothing to reject. Session state: {session.get('state')}\n")
        await safe_reply(update, "⚠️ Nothing to reject.")

import signal
import sys

def signal_handler(sig, frame):
    logger.info("🛑 KeyboardInterrupt received. Force Shutting down...")
    os._exit(0)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"❌ Exception while handling an update: {context.error}")
    # traceback.print_exception(None, context.error, context.error.__traceback__) # Optional debug
    
    # Try to notify user if possible
    if isinstance(update, Update) and update.effective_message:
        try:
            await safe_reply(update, "⚠️ A temporary network error occurred. Please try again.")
        except: pass

def main():
    # Register Signal Handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # HARDENED TIMEOUTS
    # HARDENED TIMEOUTS (Increased to 5 minutes for large file downloads)
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).connect_timeout(300).read_timeout(300).write_timeout(300).pool_timeout(300).build()
    
    app.add_error_handler(error_handler)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("getbatch", getbatch))
    app.add_handler(CommandHandler("setbatch", setbatch))
    app.add_handler(CommandHandler("compile_last", compile_last))
    app.add_handler(CommandHandler("compile_first", compile_first))
    app.add_handler(CommandHandler("approve", approve_upload))
    app.add_handler(CommandHandler("reject", reject_upload))
    app.add_handler(CommandHandler("register_promo", register_promo)) # New Command
    app.add_handler(CallbackQueryHandler(verify_watermark)) # FIXED: Register Handler
    
    # Direct Video Upload Handler
    app.add_handler(MessageHandler(filters.VIDEO | filters.Document.ALL, handle_attachment))
    
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    logger.info("🤖 Bot is running...")
    
    # Load Sessions
    load_sessions()
    
    # Check Env
    check_and_update_env()
    

    
    # Start AutoCleanup (Checks every 60 minutes, deletes files > 2 days old)
    cleanup = AutoCleanup(interval_minutes=60, age_days=2)
    cleanup.start()
    
    # Run polling
    # stop_signals=None prevents it from overwriting our signal handler (unlikely, but safe)
    app.run_polling()

# ==================== AUTO-TRAINING ====================
# ==================== AUTO-TRAINING ====================


class AutoCleanup(threading.Thread):
    def __init__(self, interval_minutes=60, age_days=2):
        super().__init__()
        self.interval = interval_minutes * 60
        self.age_days = age_days
        self.daemon = True
        self.running = True
        # Expanded Cleanup Targets
        self.target_dirs = ["downloads", "temp", "final_compilations"]
        self.state_file = "cleanup_state.json"
        self.last_run = self._load_state()

    def _load_state(self):
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_run', 0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to load cleanup state: {e}")
        return 0

    def _save_state(self):
        try:
            with open(self.state_file, 'w') as f:
                json.dump({'last_run': self.last_run}, f)
        except Exception as e:
            logger.error(f"❌ Failed to save cleanup state: {e}")

    def run(self):
        logger.info("🧹 AutoCleanup started (Persistent Mode).")
        
        while self.running:
            # Calculate time since last run
            elapsed = time.time() - self.last_run
            wait_time = max(0, self.interval - elapsed)
            
            if wait_time > 0:
                logger.info(f"⏳ Next cleanup in {int(wait_time/60)} minutes ({int(wait_time)}s)...")
                # Sleep in chunks to allow faster shutdown if needed (though daemon thread handles kill)
                # But for simplicity, simple sleep is fine as it's a daemon thread.
                time.sleep(wait_time)
            
            # Perform cleanup
            self._cleanup()
            
            # Update state
            self.last_run = time.time()
            self._save_state()
            
            # Wait for next interval (full interval now)
            # Actually, the loop logic above handles this naturally:
            # Next iteration: elapsed will be ~0, so wait_time will be ~interval.
            # So we don't need an extra sleep here.

    def _cleanup(self):
        try:
            cutoff = time.time() - (self.age_days * 86400)
            
            for target_dir in self.target_dirs:
                if not os.path.exists(target_dir):
                    continue

                for item in os.listdir(target_dir):
                    item_path = os.path.join(target_dir, item)
                    
                    if "Processed Shorts" in item or "keep" in item.lower():
                        continue
                        
                    try:
                        mtime = os.path.getmtime(item_path)
                        if mtime < cutoff:
                            if os.path.isfile(item_path):
                                os.remove(item_path)
                                logger.info(f"🗑️ Cleaned file: {item} in {target_dir}")
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path, ignore_errors=True)
                                logger.info(f"🗑️ Cleaned dir: {item} in {target_dir}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to clean {item}: {e}")
                
        except Exception as e:
            logger.error(f"❌ AutoCleanup Error: {e}")

if __name__ == '__main__':
    main()

