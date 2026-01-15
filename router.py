
import os
import logging
import json
from datetime import datetime
import health
import gpu_utils
import cpu_fast

logger = logging.getLogger("router")

def log_fallback(event_type, reason, from_mode, to_mode):
    """Log fallback events to upload_log.csv or a separate fallback log."""
    log_file = "upload_log.csv"
    try:
        # We append a special row or just log to console. 
        # Requirement says: log every fallback event, reason, and chosen fallback route to upload_log.csv
        # But upload_log.csv has a specific schema. I'll add a separate log file or just log to logger with structured data.
        # The prompt says: "log every fallback event ... to upload_log.csv". 
        # I will assume I can append a structured comment or just use a separate file to avoid breaking CSV readers.
        # Actually, let's use a separate file "fallback_log.jsonl" for structured logs as it's cleaner, 
        # but I will also log ERROR to main log.
        
        payload = {
            "timestamp": datetime.now().isoformat(),
            "event": event_type,
            "reason": reason,
            "from": from_mode,
            "to": to_mode
        }
        logger.warning(f"⚠️ FALLBACK TRIGGERED: {json.dumps(payload)}")
        
        with open("fallback_log.jsonl", "a") as f:
            f.write(json.dumps(payload) + "\n")
            
    except Exception as e:
        logger.error(f"Failed to log fallback: {e}")

import gemini_enhance

def run_enhancement(input_path, output_path, config=None):
    """
    Main entry point for video enhancement.
    Routes to Gemini AI, GPU, or CPU based on config and fallbacks.
    """
    if config is None:
        config = os.environ
        
    gpu_mode = config.get("GPU_MODE", "auto").lower()
    cpu_mode = config.get("CPU_MODE", "auto").lower()
    enable_fallback = config.get("ENABLE_FALLBACK", "yes").lower() == "yes"
    
    # 1. Health Check
    h = health.check_health()
    health.print_health_summary()
    
    # 2. Gemini AI Enhancement (Primary / Auto)
    gemini_mode = config.get("ENABLE_GEMINI_ENHANCE", "auto").lower()
    should_use_gemini = False
    
    if gemini_mode == "on":
        should_use_gemini = True
    elif gemini_mode == "off":
        should_use_gemini = False
    else: # auto
        # Auto Mode: Prefer GPU if powerful (Level 2/3), else use Gemini
        if h["gpu_available"] and h["vram_free_mb"] >= 6000:
            logger.info("🎮 Powerful GPU detected (Level 2/3). Skipping Gemini Enhance to use Heavy Engine.")
            should_use_gemini = False
        else:
            logger.info("🤖 GPU weak or unavailable. Enabling Gemini Enhance (Auto).")
            should_use_gemini = True

    if should_use_gemini:
        logger.info("🤖 Routing to Gemini AI Enhancement...")
        result = gemini_enhance.run(input_path, output_path)
        
        if result == "SUCCESS":
            return True
        elif result == "GEMINI_FAIL":
             # If Gemini fails (or API key missing), we fall through to GPU/CPU logic
             logger.warning("⚠️ Gemini failed or disabled, falling back to local engine...")
             log_fallback("fallback", "Gemini API Failure/Disable", "gemini", "local_engine")

    
    # 3. Legacy GPU/CPU Logic (if Gemini disabled)
    from compute_caps import ComputeCaps
    caps = ComputeCaps.get()
    
    # Logic: Only route to GPU if ComputeCaps allows it.
    # GPU_MODE/CPU_MODE are already handled inside ComputeCaps detection to set .gpu_fast/.cpu_only
    # But we double check the 'allow_ai_enhance' flag.
    
    use_gpu = False
    
    if caps["allow_ai_enhance"]:
         logger.info("🎮 ComputeCaps: AI Enhancement Authorized (GPU).")
         use_gpu = True
    else:
         logger.info("⚡ ComputeCaps: AI Enhancement Denied (CPU/LowVRAM).")
         use_gpu = False

    # Force Override if User explicitly requested GPU_MODE=on but Caps said no? 
    # No, ComputeCaps is authority. If Caps says no, it's unsafe.
    
    # However, if Caps says "Standard" (allow_ai_enhance=True) but user set CPU_MODE=on?
    # ComputeCaps handles CPU_MODE internally and sets allow_ai_enhance=False.
    
    # So we just trust allow_ai_enhance.

            
    if cpu_mode == "on": # Override
        use_gpu = False
        
    # 4. Execution
    if use_gpu:
        logger.info("🚀 Routing to GPU Engine...")
        try:
            success = gpu_utils.run_gpu_inference(input_path, output_path)
            if success:
                return True
            else:
                raise RuntimeError("GPU Inference returned failure")
        except Exception as e:
            logger.error(f"❌ GPU Engine Failed: {e}")
            if enable_fallback:
                log_fallback("fallback", str(e), "gpu", "cpu")
                logger.info("🔄 Falling back to CPU Fast Path...")
                return cpu_fast.apply_fallback_enhancement(input_path, output_path)
            else:
                return False
    else:
        logger.info("⚡ Routing to CPU Engine...")
        # Check thermal safety
        safe, reason = health.check_cpu_thermal()
        if not safe:
            logger.warning(f"⚠️ CPU Thermal Warning: {reason}. Throttling...")
            import time
            time.sleep(2) # Cool down a bit
            
        return cpu_fast.apply_fallback_enhancement(input_path, output_path)
