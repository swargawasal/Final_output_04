"""
Import Gate Authority
---------------------
Centralized gatekeeper for heavy imports (Torch, Diffusers, etc.).
Consults ComputeCaps to allow or block modules based on VRAM availability.

Usage:
    from import_gate import ImportGate
    
    # Safe Import
    torch = ImportGate.get("torch") # Returns module or None
    
    if torch:
        # Use GPU code
    else:
        # Fallback to CPU/OpenCV
"""

import sys
import logging
import importlib
from compute_caps import ComputeCaps

logger = logging.getLogger("import_gate")

class ImportGate:
    _loaded_modules = {}
    
    # Modules classified as "HEAVY" (Require GPU_ENHANCED_MODE)
    _heavy_registry = {
        "torch": "Per-frame AI / Tensors",
        "diffusers": "Texture Synthesis / Stable Diffusion",
        "transformers": "Advanced NLP/Vision",
        "basicsr": "Super Resolution",
        "gfpgan": "Face Restoration",
        "realesrgan": "Upscaling"
    }

    @staticmethod
    def get(module_name: str):
        """
        Attempts to import a module respecting Compute Caps.
        Returns: Module object if allowed/successful, else None.
        """
        # 0. Check Cache
        if module_name in ImportGate._loaded_modules:
            return ImportGate._loaded_modules[module_name]

        # 1. Check Compute Caps
        caps = ComputeCaps.get()
        allow_ai = caps.get("allow_ai_enhance", False)
        
        # 2. Check Registry
        if module_name in ImportGate._heavy_registry:
            if not allow_ai:
                # BLOCKED
                logger.info(f"🚫 ImportGate: Blocking '{module_name}' (CPU Safe Mode Active).")
                logger.debug(f"   └─ Reason: {ImportGate._heavy_registry[module_name]}")
                return None
            else:
                # ALLOWED: But log it
                logger.info(f"⚡ ImportGate: Allowing '{module_name}' (GPU Enhanced Mode).")

        # 3. Attempt Import
        try:
            mod = importlib.import_module(module_name)
            ImportGate._loaded_modules[module_name] = mod
            return mod
        except ImportError as e:
            logger.warning(f"⚠️ ImportGate: Failed to import '{module_name}': {e}")
            return None
        except Exception as e:
            logger.error(f"❌ ImportGate: Unexpected error importing '{module_name}': {e}")
            return None

    @staticmethod
    def is_active(module_name: str) -> bool:
        """Returns True if module is successfully loaded and active."""
        return module_name in sys.modules and ImportGate.get(module_name) is not None
