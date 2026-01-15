
import os
import sys
import requests
import logging
import shutil
import subprocess
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] tools-install: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("tools-install")

# Directories
BASE_DIR = os.getcwd()
MODELS_ROOT = os.path.join(BASE_DIR, "models")
MODELS_HEAVY_DIR = os.path.join(MODELS_ROOT, "heavy")

os.makedirs(MODELS_ROOT, exist_ok=True)
os.makedirs(MODELS_HEAVY_DIR, exist_ok=True)

# Model Definitions
MODELS = {
    "deploy.prototxt": {
        "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "dest_folder": MODELS_ROOT,
        "min_size": 1 * 1024, # ~1KB
        "type": "essential"
    },
    "res10_300x300_ssd_iter_140000.caffemodel": {
        "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "dest_folder": MODELS_ROOT,
        "min_size": 5 * 1024 * 1024, # ~10MB
        "type": "essential"
    },
    "RealESRGAN_x4plus.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "dest_folder": MODELS_HEAVY_DIR,
        "min_size": 10 * 1024 * 1024,
        "type": "heavy"
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
        "dest_folder": MODELS_HEAVY_DIR,
        "min_size": 10 * 1024 * 1024,
        "type": "heavy"
    },
    "parsing_parsenet.pth": {
        "url": "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth",
        "dest_folder": MODELS_HEAVY_DIR,
        "min_size": 10 * 1024 * 1024,
        "type": "heavy"
    }
}

MIN_DISK_SPACE_MB = 2000 # 2GB required to be safe
MIN_VRAM_MB = 6000 # 6GB VRAM required for heavy models

def create_retrying_session(retries=3, backoff_factor=1.0):
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=(500, 502, 503, 504),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def check_disk_space(path):
    """Ensure sufficient disk space available."""
    try:
        total, used, free = shutil.disk_usage(path)
        free_mb = free / (1024 * 1024)
        if free_mb < MIN_DISK_SPACE_MB:
            logger.error(f"❌ Low Disk Space: {free_mb:.1f}MB available, need {MIN_DISK_SPACE_MB}MB.")
            return False
        return True
    except Exception as e:
        logger.warning(f"⚠️ Could not verify disk space: {e}")
        return True 

def clean_orphaned_parts():
    """Cleanup .part files from previous failed runs."""
    try:
        # Scan both directories
        for d in [MODELS_ROOT, MODELS_HEAVY_DIR]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    if f.endswith(".part"):
                        path = os.path.join(d, f)
                        logger.info(f"🧹 Removing orphaned partial download: {f}")
                        os.remove(path)
    except Exception: pass

def get_gpu_vram():
    """
    Returns GPU VRAM in MB if NVIDIA GPU is available, else 0.
    Uses nvidia-smi.
    """
    if not shutil.which("nvidia-smi"):
        return 0
        
    try:
        # nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
        # Output example: 12288
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"], 
            check=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=3,
            text=True
        )
        output = result.stdout.strip()
        # Handle multiple GPUs - take the max VRAM? Or the first?
        # Usually first is primary.
        lines = output.split('\n')
        if not lines: return 0
        
        vram_mb = int(lines[0].strip())
        logger.info(f"🔍 GPU Detected. VRAM: {vram_mb} MB")
        return vram_mb
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
        logger.warning(f"⚠️ Failed to query VRAM via nvidia-smi: {e}")
        return 0
    except Exception as e:
        logger.warning(f"⚠️ GPU Check Error: {e}")
        return 0

def download_file(url, dest_folder, min_size, session):
    dest_path = os.path.join(dest_folder, os.path.basename(url))

    if os.path.exists(dest_path):
        # Integrity check
        if os.path.getsize(dest_path) > min_size:
            logger.info(f"✅ {os.path.basename(dest_path)} validation OK.")
            return True
        else:
            logger.warning(f"⚠️ {os.path.basename(dest_path)} is invalid/small. Re-downloading.")
            try:
                os.remove(dest_path)
            except OSError:
                pass

    logger.info(f"📥 Downloading {os.path.basename(dest_path)}...")
    temp_path = dest_path + ".part"
    
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    try:
        response = session.get(url, stream=True, timeout=(10, 30))
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        
        if total_size > 0 and total_size < min_size: 
             logger.error(f"❌ Server returned invalid file size ({total_size} bytes). Aborting.")
             return False

        with open(temp_path, 'wb') as f, tqdm(
            desc=os.path.basename(dest_path),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            disable=None 
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = f.write(data)
                bar.update(size)
        
        if os.path.getsize(temp_path) < min_size:
             raise ValueError("Downloaded file too small (Corrupt/HTML?)")

        shutil.move(temp_path, dest_path)
        logger.info(f"✨ Installed {os.path.basename(dest_path)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to download {os.path.basename(dest_path)}: {e}")
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False

def main():
    logger.info("🚀 Starting Smart Tools Installer (VRAM Aware)...")
    
    # 0. Cleanup
    clean_orphaned_parts()

    # 1. Check & Config
    fast_mode = os.getenv("FAST_MODE", os.getenv("AI_FAST_MODE", "no")).lower() == "yes"
    install_heavy = os.getenv("INSTALL_HEAVY", "auto").lower() # Default to auto
    
    vram_mb = get_gpu_vram()
    
    download_heavy = False
    
    if fast_mode:
        logger.info("⚡ FAST_MODE enabled. Skipping HEAVY models.")
        download_heavy = False
    elif install_heavy == "yes":
        logger.info("⚠️ INSTALL_HEAVY=yes forced. Ignoring VRAM check.")
        download_heavy = True
    elif install_heavy == "no":
        logger.info("ℹ️ INSTALL_HEAVY=no. Skipping heavy models.")
        download_heavy = False
    else:
        # Auto Mode - Strict VRAM Check
        if vram_mb > MIN_VRAM_MB:
            logger.info(f"✅ Sufficient VRAM ({vram_mb}MB > {MIN_VRAM_MB}MB). Enabling HEAVY models.")
            download_heavy = True
        else:
            if vram_mb > 0:
                logger.warning(f"⚠️ Insufficient VRAM ({vram_mb}MB < {MIN_VRAM_MB}MB). Skipping heavy models to prevent OOM.")
            else:
                logger.warning("⚠️ No NVIDIA GPU / VRAM detected. Skipping heavy models.")
            download_heavy = False

    # 2. Disk Checks
    if not check_disk_space(MODELS_ROOT):
        return

    # 3. Download Loop
    session = create_retrying_session()
    
    total_tasks = 0
    success_tasks = 0

    for name, config in MODELS.items():
        is_heavy = config['type'] == 'heavy'
        
        # Determine if we should download this file
        should_download = False
        if is_heavy:
            if download_heavy: should_download = True
        else:
            # Essential files always download (unless really critical disk failure)
            should_download = True
            
        if should_download:
            total_tasks += 1
            if download_file(config['url'], config['dest_folder'], config['min_size'], session):
                success_tasks += 1

    logger.info("---------------------------------------------------")
    if success_tasks == total_tasks:
        logger.info(f"✨ Success! {success_tasks}/{total_tasks} models ready.")
    else:
        logger.warning(f"⚠️ Completed with warnings: {success_tasks}/{total_tasks} models ready.")
    logger.info("---------------------------------------------------")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.error("\n🛑 Installer interrupted by user.")
        clean_orphaned_parts()
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Installer Crash: {e}", exc_info=True)
        sys.exit(1)
