"""
Compute Capability Authority
----------------------------
Standardizes hardware detection across the bot.
Authorized Source of Truth for:
- CUDA Availability
- VRAM Capacity
- CPU-Fast Mode Enforcement

Rule: Calculated ONCE at startup. Reused everywhere.
"""

import os
import sys
import logging
import subprocess
import shutil

logger = logging.getLogger("compute_caps")

class ComputeCaps:
    _instance = None
    _caps = {
        "has_cuda": False,
        "vram_gb": 0.0,
        "gpu_fast": False,
        "cpu_only": True,
        "allow_ai_enhance": False
    }
    _initialized = False

    @classmethod
    def get(cls):
        if not cls._initialized:
            cls._detect()
        return cls._caps

    @classmethod
    def _detect(cls):
        """
        Detects hardware capabilities WITHOUT importing torch/tensorflow if possible initially.
        Uses nvidia-smi as primary, falls back to torch if absolutely needed but wary of import cost.
        Actually, to be 100% accurate for 'torch' usage, we might need torch, BUT 
        we want to avoid importing it if we are in CPU_FAST mode.
        """
        config = os.environ
        
        # 1. User Override (CPU_ONLY)
        if config.get("COMPUTE_MODE", "auto").lower() == "cpu":
            logger.info("⚙️ ComputeCaps: CPU Mode Forced via Env.")
            cls._set_cpu_only()
            return
            
        # 2. Nvidia-SMI Check (Lightweight)
        has_nvidia = False
        vram = 0.0
        try:
            # Check for nvidia-smi
            smi = shutil.which("nvidia-smi")
            if smi:
                # Query memory
                cmd = [smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"]
                result = subprocess.check_output(cmd, encoding='utf-8')
                # Parse first GPU
                lines = result.strip().split('\n')
                if lines:
                    vram_mb = float(lines[0])
                    vram = vram_mb / 1024.0
                    has_nvidia = True
        except Exception as e:
            logger.debug(f"ComputeCaps: nvidia-smi check failed: {e}")
            has_nvidia = False
            
        # 3. Decision
        if has_nvidia:
            # We found a GPU via system tools.
            # However, PyTorch might not be installed with CUDA support.
            # This is the tricky part. If we rely ONLY on smi, we might try to import torch and fail.
            # But the goal is to NOT import torch if we don't have a GPU.
            # If we DO have a GPU, we are authorized to spend time importing torch to confirm.
            
            logger.info(f"⚙️ ComputeCaps: GPU Detected (VRAM={vram:.1f}GB). Verifying Torch...")
            try:
                import torch
                if torch.cuda.is_available():
                     cls._caps["has_cuda"] = True
                     cls._caps["vram_gb"] = vram
                     cls._caps["cpu_only"] = False
                     
                     # 4. Capability Tier
                     if vram >= 6.0:
                         cls._caps["gpu_fast"] = True
                         cls._caps["allow_ai_enhance"] = True
                         logger.info("   └─ Status: PRO (GPU Fast Path Enabled)")
                     elif vram >= 4.0:
                         cls._caps["gpu_fast"] = False # Safe Mode
                         cls._caps["allow_ai_enhance"] = True
                         logger.info("   └─ Status: STANDARD (GPU Safe Mode)")
                     else:
                         cls._caps["gpu_fast"] = False
                         cls._caps["allow_ai_enhance"] = False # < 4GB too risky for Heavy AI
                         logger.info("   └─ Status: LOW_RES (GPU Present but VRAM constrained)")
                else:
                    logger.warning("   └─ Warning: Hardware detected but Torch CUDA is unresponsive.")
                    cls._set_cpu_only()
                    
            except ImportError:
                logger.warning("   └─ Warning: GPU present but 'torch' not found.")
                cls._set_cpu_only()
                
        else:
             logger.info("⚙️ ComputeCaps: No GPU detected. Defaulting to CPU.")
             cls._set_cpu_only()

        cls._initialized = True

    @classmethod
    def _set_cpu_only(cls):
        cls._caps["has_cuda"] = False
        cls._caps["vram_gb"] = 0.0
        cls._caps["gpu_fast"] = False
        cls._caps["cpu_only"] = True
        cls._caps["allow_ai_enhance"] = False # CPU too slow for heavy AI
