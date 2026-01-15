# ============================================================
# GOOGLE COLAB HARDENED INSTALLATION SCRIPT
# PRODUCTION-GRADE • AUTO-HEALING • ISOLATED
# ============================================================

import subprocess
import sys
import os
import shutil
import time

# --- CONFIGURATION ---
VENV_DIR = "venv"
IS_WINDOWS = os.name == 'nt'
VENV_BIN = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin")
# In standard venvs, the executable is always 'python' (or python.exe), never 'python3'
PYTHON_EXEC = os.path.join(VENV_BIN, "python" + (".exe" if IS_WINDOWS else ""))
PIP_EXEC = os.path.join(VENV_BIN, "pip" + (".exe" if IS_WINDOWS else ""))

# HEALING CANDIDATES FOR NUMPY
# Order matters: Newest compatible -> Oldest supported
NUMPY_CANDIDATES = [
    "1.26.4",
    "1.25.2",
    "1.24.4",
    "1.23.5"
]

def log(msg, type="INFO"):
    icons = {"INFO": "ℹ️", "SUCCESS": "✅", "WARN": "⚠️", "ERROR": "❌", "ACTION": "🚀"}
    icon = icons.get(type, "🔹")
    print(f"{icon} [{type}] {msg}")

def run_cmd(cmd, desc=None, check=True, use_shell=False):
    if desc:
        log(f"{desc}...", "ACTION")
    
    try:
        if not use_shell and isinstance(cmd, str):
            cmd_list = cmd.split()
        else:
            cmd_list = cmd

        result = subprocess.run(
            cmd_list, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            shell=use_shell
        )
        if desc:
            log(f"Completed: {desc}", "SUCCESS")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        if desc:
            log(f"Failed: {desc}", "ERROR")
            print(f"   Error Output: {e.stderr}")
        if check:
            raise e
        return False, e.stderr

def ensure_venv():
    if not os.path.exists(VENV_DIR):
        log(f"Creating virtual environment in '{VENV_DIR}'...", "ACTION")
        subprocess.run([sys.executable, "-m", "venv", VENV_DIR], check=True)
        log("Virtual environment created.", "SUCCESS")
    else:
        log(f"Virtual environment '{VENV_DIR}' already exists.", "INFO")

def install_deps():
    # 1. System Deps (FFmpeg)
    if shutil.which("ffmpeg") is None:
        try:
            # APT-GET requires shell=True for && chaining
            run_cmd("apt-get update && apt-get install -y ffmpeg", "Installing FFmpeg (System)", check=False, use_shell=True)
        except Exception:
             log("Could not install FFmpeg via apt-get. Ensure it is installed manually.", "WARN")
    
    # 2. Upgrade pip in venv
    run_cmd([PYTHON_EXEC, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip")

    # 3. Install Python Deps
    req_file = "requirements_colab.txt" if os.path.exists("requirements_colab.txt") else "requirements.txt"
    if os.path.exists(req_file):
        log(f"Installing dependencies from {req_file}...", "ACTION")
        run_cmd([PIP_EXEC, "install", "-r", req_file], "Dependency Installation")
    else:
        log(f"{req_file} not found!", "ERROR")
        sys.exit(1)

def check_ABI_health():
    """
    Validates if cv2 and numpy can coexist.
    Captures both ImportError and ABI-level crashes.
    """
    check_script = "import cv2; import numpy; print('Imports Successful')"
    try:
        subprocess.run(
            [PYTHON_EXEC, "-c", check_script], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        return True
    except subprocess.CalledProcessError:
        return False

def reinstall_opencv():
    log("♻️ Attempting OpenCV Reinstall...", "ACTION")
    pkgs = ["opencv-python", "opencv-python-headless", "opencv-contrib-python"]
    
    # Force clean uninstall
    for pkg in pkgs:
        run_cmd([PIP_EXEC, "uninstall", "-y", pkg], check=False)
        
    # Reinstall headless (best for server/colab)
    run_cmd([PIP_EXEC, "install", "opencv-python-headless"], "Reinstalling OpenCV-Headless")

def auto_heal_numpy(retry_opencv_fallback=True):
    log("Validating NumPy <-> OpenCV compatibility...", "ACTION")
    
    if check_ABI_health():
        log("Environment is healthy. No repairs needed.", "SUCCESS")
        return True

    log("Detected NumPy/OpenCV ABI mismatch or import error. Starting Auto-Heal...", "WARN")
    
    for version in NUMPY_CANDIDATES:
        log(f"Attempting repair with NumPy {version}...", "ACTION")
        
        run_cmd([PIP_EXEC, "uninstall", "-y", "numpy"], check=False)
        
        try:
            run_cmd([PIP_EXEC, "install", f"numpy=={version}"], f"Installing NumPy {version}")
        except Exception:
             continue
            
        if check_ABI_health():
            log(f"✅ Auto-Heal Successful! Stable NumPy version: {version}", "SUCCESS")
            # Freeze working config
            with open("requirements.lock", "w") as f:
                subprocess.run([PIP_EXEC, "freeze"], stdout=f, text=True)
            return True
        else:
            log(f"NumPy {version} failed verification.", "WARN")
            
    # If all NumPy versions fail
    if retry_opencv_fallback:
        log("⚠️ All NumPy versions failed. Triggering OpenCV Reinstall logic...", "WARN")
        reinstall_opencv()
        # Retry healing ONCE more with fresh OpenCV
        return auto_heal_numpy(retry_opencv_fallback=False)

    log("❌ CRITICAL: Auto-healing failed after OpenCV reinstall.", "ERROR")
    sys.exit(1)

def patch_basicsr():
    log("Checking for basicsr compatibility patches...", "INFO")
    try:
        res = subprocess.check_output(
            [PYTHON_EXEC, "-c", "import site; print(site.getsitepackages()[0])"], 
            text=True
        ).strip()
        site_pkg = res
    except:
        return

    target_file = os.path.join(site_pkg, "basicsr", "data", "degradations.py")
    
    if os.path.exists(target_file):
        with open(target_file, "r") as f:
            content = f.read()
        
        old_import = "from torchvision.transforms.functional_tensor import rgb_to_grayscale"
        new_import = "from torchvision.transforms.functional import rgb_to_grayscale"
        
        if old_import in content:
            new_content = content.replace(old_import, new_import)
            with open(target_file, "w") as f:
                f.write(new_content)
            log("Patched basicsr/data/degradations.py (torchvision fix)", "SUCCESS")

def verify_and_report():
    log("Final Environment Verification...", "ACTION")
    
    verify_script = """
import sys
try:
    import numpy
    print(f"NumPy: {numpy.__version__}")
except:
    print("NumPy: FAILED")

try:
    import cv2
    print(f"OpenCV: {cv2.__version__}")
except:
    print("OpenCV: FAILED")

try:
    import torch
    print(f"Torch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    print("Torch: WARN - Not found or Error (Check logs)")
"""
    try:
        result = subprocess.run(
            [PYTHON_EXEC, "-c", verify_script], 
            check=True, 
            capture_output=True, 
            text=True
        )
        
        output = result.stdout.strip()
        print("\n" + "="*40)
        print("🎉 ENVIRONMENT REPORT")
        print("="*40)
        print(output)
        print("="*40 + "\n")
        
        if "FAILED" in output:
             log("Verification warnings detected. Check output above.", "WARN")
        else:
             log("Installation Complete! Ready to launch.", "SUCCESS")
             print(f"\n    !{PYTHON_EXEC} main.py\n")
        
    except subprocess.CalledProcessError:
        log("Verification script crashed.", "ERROR")

def main():
    print("\n🔒 STARTING HARDENED COLAB INSTALLATION (FINAL)\n")
    
    ensure_venv()
    install_deps()
    auto_heal_numpy()
    patch_basicsr()
    
    if os.path.exists("tools-install.py"):
        log("Running tools-install.py...", "ACTION")
        run_cmd([PYTHON_EXEC, "tools-install.py"], "Installing Heavy Models", check=False)
    
    verify_and_report()

if __name__ == "__main__":
    main()
