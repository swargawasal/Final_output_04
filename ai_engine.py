# ai_engine.py - OPTIMIZED AI ENHANCEMENT ENGINE
import os
import logging
import cv2

import numpy as np
import time
from tqdm import tqdm
from compute_caps import ComputeCaps

# Lazy Imports (Type hinting placeholders if needed, or just rely on runtime)
torch = None
RRDBNet = None
RealESRGANer = None
GFPGANer = None

logger = logging.getLogger("ai_engine")

class HeavyEditor:
    """AI Video Enhancement Engine using RealESRGAN + GFPGAN (Optimized & governed)"""
    def __init__(self, model_dir="models/heavy", scale=2, face_enhance=True):
        self.model_dir = model_dir
        self.scale = scale
        self.face_enhance = face_enhance
        
        # 1. Compute Authority Check
        self.caps = ComputeCaps.get()
        self.device = 'cpu' # Default

        
        # Paths to models
        self.realesrgan_x4_path = os.path.join(model_dir, 'RealESRGAN_x4plus.pth')
        self.realesrgan_x2_path = os.path.join(model_dir, 'RealESRGAN_x2plus.pth')
        self.gfpgan_model_path = os.path.join(model_dir, 'GFPGANv1.4.pth')
        
        self.face_enhancer = None
        self.upsampler = None
        self._model_broken = False
        
        # 1. Antigravity Safety Layers (Metadata & Debug)
        self._meta = {
           "gpu_type": "Unknown",
           "vram_gb": 0.0,
           "mode": "INIT",
           "auto_scale_source": "manual",
           "enhancement_level": scale,
           "face_enhance": face_enhance
        }
        self._debug = {"errors": []}
        
        # 12. Silent Nondestructive Fallback
        try:
            self._load_models()
        except Exception as e:
            logger.error(f"❌ Critical AI Engine Failure: {e}. Falling back to Neutral Pass-Through Mode.")
            self._disable_enhancement("Init Crash")

    def _disable_enhancement(self, reason: str):
        """Cleanly disable all enhancement features."""
        logger.warning(f"⚠️ Disabling AI Enhancement: {reason}")
        self.scale = 1
        self.face_enhance = False
        self.upsampler = None
        self.face_enhancer = None
        self._model_broken = True
        self._meta["mode"] = "DISABLED"

    def _ensure_model(self, path, url):
        """Check if model exists, if not download it (Sanity Wrapper)."""
        if self._model_broken: return
        
        # 11. CPU / Safety Check
        if self.device.type != 'cuda':
            logger.warning(f"⚠️ CPU detected. Skipping auto-download of {path}.")
            return

        if not os.path.exists(path):
            logger.info(f"⬇️ Model not found: {path}. Downloading from {url}...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                torch.hub.download_url_to_file(url, path)
                logger.info(f"✅ Downloaded: {path}")
            except Exception as e:
                logger.error(f"❌ Failed to download model: {e}")
                if os.path.exists(path): # Cleanup partial download
                    try: os.remove(path)
                    except: pass
                # Soft failure: don't crash, just mark broken
                self._disable_enhancement("Download Failure")

    def _get_device_config(self) -> dict:
        """
        2. Strict GPU Memory Governor (PURE FUNCTION).
        Returns a configuration dictionary. Does NOT mutate state.
        """
        config = {
            "tile": 0,
            "half": False,
            "face_enhance": True, # Default, governed below
            "device": self.device,
            "auto_scale": 2,
            "meta_update": {} # Helper to pass meta back
        }
        
        # 3. Process Governor (CPU Safety via ComputeCaps)
        if self.caps["cpu_only"] or not self.caps["allow_ai_enhance"]:
            config["tile"] = 100 # Aggressive tiling for CPU
            config["half"] = False
            config["face_enhance"] = False # Strictly disable face enhance on CPU
            config["auto_scale"] = 1 # Neutral scale preferred
            config["meta_update"] = {"mode": "CPU_BASIC" if self.caps["cpu_only"] else "GPU_LOW_VRAM"}
            # Ensure device is set to CPU just in case
            config["device"] = 'cpu'
            return config
            
        # GPU Mode Checks (We know we are allowed to use AI, so we can import torch)
        global torch
        if torch is None: import torch 
        
        # Use actual device from torch if available
        config["device"] = torch.device('cuda' if self.caps["has_cuda"] else 'cpu')

        try:
            # We trust ComputeCaps for VRAM, but strict check here doesn't hurt if we have torch
            vram = self.caps["vram_gb"]
            name = "GPU" # Placeholder or get from torch if needed
            
            config["meta_update"] = {
                "gpu_type": name,
                "vram_gb": vram,
                "mode": "GPU_HEAVY"
            }
            
            # Env Override
            if os.getenv("FORCE_HEAVY_MODE", "no").lower() == "yes":
                logger.warning("⚠️ FORCE_HEAVY_MODE enabled. Bypassing VRAM checks.")
                config["half"] = True
                config["tile"] = 0
                config["auto_scale"] = 4
                return config

            # 2. VRAM Thresholds
            if vram < 4.0:
                 # Level 0: < 4GB -> STRICT DISABLE
                 # Even 4-6GB is risky for RealESRGAN+GFPGAN, but we allow 6GB below.
                 config["face_enhance"] = False
                 config["auto_scale"] = 1 
                 config["meta_update"]["mode"] = "GPU_INSUFFICIENT"
                 
            elif vram < 6.0:
                # Level 1: 4-6GB -> Force Pass-Through (Logic request: <6GB disable heavy AI)
                config["face_enhance"] = False
                config["auto_scale"] = 1 
                config["meta_update"]["mode"] = "GPU_DISABLED_LOW_VRAM"
                
            elif vram < 8.0:
                # Level 2: 6-8GB -> Stable Mode
                config["tile"] = 300
                config["half"] = False # Stability over speed
                
                # Check explicit force face
                if os.getenv("FORCE_FACE", "no").lower() != "yes":
                    config["face_enhance"] = False
                
                config["auto_scale"] = 2
                
            elif vram < 12.0:
                 # Level 3: 8-12GB -> High Perf
                 config["tile"] = 400
                 config["half"] = True
                 config["auto_scale"] = 2
                 
            else:
                 # Level 4: > 12GB -> Ultra
                 config["tile"] = 0
                 config["half"] = True
                 config["auto_scale"] = 4

        except Exception as e:
            logger.warning(f"⚠️ VRAM detection failed: {e}. Defaulting to Safe-Mode.")
            config.update({"tile": 100, "face_enhance": False, "auto_scale": 1, "half": False})
            config["meta_update"] = {"mode": "GPU_ERROR_FALLBACK"}
            
        return config

    def _load_models(self):
        """Atomic Model Loading with Governor Check."""
        if self._model_broken: return

        # 0. Lazy Import Gate
        if self.caps.get("allow_ai_enhance", False):
            try:
                global torch, RRDBNet, RealESRGANer, GFPGANer
                if torch is None:
                    import torch
                    from basicsr.archs.rrdbnet_arch import RRDBNet
                    from realesrgan import RealESRGANer
                    from gfpgan import GFPGANer
            except ImportError as e:
                logger.error(f"❌ Critical Import Failure (Torch/RealESRGAN): {e}")
                self._disable_enhancement("Missing Dependencies")
                return

        try:
            # 1. Get Pure Config
            config = self._get_device_config()
            
            # 2. Apply Meta Updates
            if "meta_update" in config:
                self._meta.update(config["meta_update"])
            
            # 3. Check Governor Decision
            if config.get("auto_scale") == 1:
                 # Governor requested pass-through
                 logger.info(f"🛡️ AI Governor: Enhancement disabled (Mode: {self._meta.get('mode')}).")
                 self.scale = 1
                 self.face_enhance = False
                 return # Exit safely, keeping scale=1
            
            # 4. Resolve Env Scale vs VRAM Scale
            enhancement_env = os.getenv("ENHANCEMENT_LEVEL", "2x").lower()
            if enhancement_env == "auto":
                self.scale = config["auto_scale"]
                logger.info(f"🧠 Auto-scale resolved: scale={self.scale}")
            else:
                try: 
                    # Attempt to parse user intent '4x' -> 4
                    if "4" in enhancement_env: self.scale = 4
                    elif "2" in enhancement_env: self.scale = 2
                    else: self.scale = 2 # Default safe
                except: self.scale = 2
                
            # 5. Face Enhance Decision
            if not config["face_enhance"]:
                if self.face_enhance: logger.info("🔧 Face enhancement disabled by Governor (VRAM).")
                self.face_enhance = False

            # 6. Load Upsampler
            logger.info(f"🚀 Loading RealESRGAN (x{self.scale}) on {self.device}...")
            
            if self.scale <= 2:
                model_path = self.realesrgan_x2_path
                model_scale = 2
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
            else:
                model_path = self.realesrgan_x4_path
                model_scale = 4
                url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                
            self._ensure_model(model_path, url)
            if self._model_broken: return # Check after ensure

            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=model_scale)
            
            self.upsampler = RealESRGANer(
                scale=model_scale,
                model_path=model_path,
                model=model,
                tile=config["tile"],
                tile_pad=10,
                pre_pad=0,
                half=config["half"],
                device=self.device
            )
            
            # 7. Load Face Enhancer
            if self.face_enhance:
                self._ensure_model(self.gfpgan_model_path, "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth")
                if not self._model_broken:
                     self.face_enhancer = GFPGANer(
                        model_path=self.gfpgan_model_path,
                        upscale=self.scale,
                        arch='clean',
                        channel_multiplier=2,
                        bg_upsampler=self.upsampler
                    )
                     logger.info("✅ GFPGAN Loaded.")

        except Exception as e:
            logger.error(f"❌ Model Init Crashed: {e}")
            self._disable_enhancement("Load Exception")

    def enhance_frame(self, img):
        """Enhance a single frame. Safe, deterministic, strictly typed."""
        # 5. Frame-Safe Failover
        try:
            # Type and Validity Check
            if img is None or img.size == 0:
                return img
                
            h, w = img.shape[:2]
            target_h, target_w = h * self.scale, w * self.scale
                
            # Passthrough Conditions
            if self._model_broken or self.scale == 1 or not self.upsampler:
                return img
            
            # Inference
            with torch.no_grad():
                if self.face_enhance and self.face_enhancer:
                    _, _, output = self.face_enhancer.enhance(
                        img, 
                        has_aligned=False, 
                        only_center_face=False, 
                        paste_back=True
                    )
                else:
                    output, _ = self.upsampler.enhance(img, outscale=self.scale)
            
            # 9. Pipeline Integrity Check & Dtype enforcement
            if output is None:
                raise ValueError("Model returned None")
                
            # If float, convert to uint8 safely
            if output.dtype != np.uint8:
                 if output.dtype == np.float32 or output.dtype == np.float64:
                      output = (output * 255.0).clip(0, 255).astype(np.uint8)
                 else:
                      output = output.astype(np.uint8) # Blind cast fallback
            
            # 2. RESOLUTION CONTRACT (CRITICAL)
            # Ensure output matches expected scale EXACTLY
            oh, ow = output.shape[:2]
            if oh != target_h or ow != target_w:
                # Mismatch detected (FaceEnhancer sometimes shifts size by pixels)
                output = cv2.resize(output, (target_w, target_h))
            
            # Skin Protection (Conditional)
            if os.getenv("ENABLE_SKIN_PROTECT", "yes").lower() == "yes":
                output = self._protect_skin(output)

            return output
            
        except Exception as e:
            # 10. Error Channel & Per-Frame Fallback
            if len(self._debug["errors"]) < 10:
                self._debug["errors"].append(str(e))
            
            # CRITICAL FALLBACK: Resize original to target resolution
            # If we simply return img (1x), and the writer expects 2x/4x, it WILL crash
            if self.scale > 1:
                return cv2.resize(img, (int(w * self.scale), int(h * self.scale)))
            return img

    def _protect_skin(self, img_bgr):
        """Simple skin brightness clamping."""
        try:
            skin_max = int(os.getenv("SKIN_MAX_BRIGHTNESS", 175))
            img_ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
            # Range: Cr[133,173], Cb[77,127]
            mask = cv2.inRange(img_ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
            if cv2.countNonZero(mask) == 0: return img_bgr
            
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Mask bright skin
            bright = cv2.threshold(l, skin_max, 255, cv2.THRESH_BINARY)[1]
            combined = cv2.bitwise_and(bright, mask)
            
            if cv2.countNonZero(combined) > 0:
                l = np.where(combined > 0, skin_max, l)
                return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            return img_bgr
        except:
             return img_bgr

    def process_video(self, input_path, output_path, progress_callback=None):
        logger.info(f"🎬 Starting Video Enhancement: {input_path}")
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logger.error("❌ Could not open input video.")
            return False

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 8. Writer Safety: Late Init
        # We trust self.scale is stable now (set in __init__/_load_models)
        target_width = int(width * self.scale)
        target_height = int(height * self.scale)
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if fps > 120: fps = 30 # Sanity clamp
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (target_width, target_height))
        
        if not writer.isOpened():
             logger.error("❌ Failed to open video writer.")
             cap.release()
             return False
             
        # 7. Batch Logic
        # Heuristic: 1 batch per 4GB VRAM, capped at 4. CPU = 1 (Strict)
        batch_size = 1
        if self._model_broken:
             batch_size = 8 # Fast CPU copy
        elif self.device.type == 'cuda':
             vram = self._meta.get("vram_gb", 0)
             batch_size = min(max(1, int(vram // 4)), 4)
             
        frame_buffer = []
        processed_count = 0
        
        try:
            for i in tqdm(range(total_frames), desc="Enhancing"):
                ret, frame = cap.read()
                if not ret: break
                
                # Check stream integrity
                if frame.shape[:2] != (height, width):
                     frame = cv2.resize(frame, (width, height))
                
                frame_buffer.append(frame)
                
                if len(frame_buffer) >= batch_size:
                    enhanced_batch = self._process_batch(frame_buffer)
                    for eff in enhanced_batch:
                         # Assertion: Resolution Contract
                         # If enhance_frame failed, it returned scaled image.
                         # If it worked, it returned scaled image.
                         # We verified this in enhance_frame logic.
                         if eff.shape[:2] != (target_height, target_width):
                              # Last ditch panic resize
                              eff = cv2.resize(eff, (target_width, target_height))
                         writer.write(eff)
                         processed_count += 1
                    frame_buffer = []
                    
                if progress_callback and i % 10 == 0:
                     progress_callback(i / total_frames)
                     
            # Flush
            if frame_buffer:
                enhanced_batch = self._process_batch(frame_buffer)
                for eff in enhanced_batch:
                     if eff.shape[:2] != (target_height, target_width):
                          eff = cv2.resize(eff, (target_width, target_height))
                     writer.write(eff)
                     processed_count += 1
                     
            return True
            
        except Exception as e:
            logger.error(f"❌ Video processing crashed: {e}")
            return False
        finally:
            cap.release()
            writer.release()
            
    def _process_batch(self, frames):
        """Process batch. Returns list of enhanced frames."""
        # Note: We process sequentially here because RealESRGANer isn't natively batch-optimized 
        # in this wrapper, but keeping the structure allows future optimization or GPU queuing.
        # It also keeps the logic clean.
        out = []
        for f in frames:
            out.append(self.enhance_frame(f))
        return out
