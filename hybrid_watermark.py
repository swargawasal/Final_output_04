"""
Hybrid Watermark Manager (Gemini Authority)
-------------------------------------------
STRICT MODE:
- Gemini Vision is the ONLY detection source.
- No ML, No OpenCV Detection.
- If Gemini fails/quota -> Status: CLEAN (Skip).
- Geometric Hardening & Face Safety Enforced.
"""

import os
import json
import logging
import cv2
import numpy as np
import time
import uuid
import shutil
from import_gate import ImportGate

# Import FaceProtector for strict safety overrides
try:
    from opencv_watermark import FaceProtector
except ImportError:
    # Fail-safe mock if import fails (should not happen in prod)
    class FaceProtector:
        @staticmethod
        def is_safe_region(frame, box): return True, "Safe"

logger = logging.getLogger("hybrid_watermark")

class HybridWatermarkDetector:
    def __init__(self):
        self.session_blacklist = {} 
        self.removability_cache = {}

    def _error_json(self, msg):
        return json.dumps({
            "watermarks": [],
            "count": 0,
            "status": "ERROR",
            "context": {"error": msg}
        })



    def confirm_learning(self, context: dict, is_positive: bool):
        """
        Logs user feedback (Reinforcement Learning Stub).
        in "Strict Mode", this just logs the failure to influence the next retry's prompt.
        """
        try:
             # In future, this could update a database or weight file.
             # For now, we rely on 'retry_level' to escalate.
             feedback = "POSITIVE" if is_positive else "NEGATIVE"
             logger.info(f"🧠 HybridWatermark Learning: Received {feedback} feedback.")
             if not is_positive:
                 logger.info("   └─ Will trigger deeper scan on next retry.")
        except Exception as e:
            logger.warning(f"Learning feedback failed: {e}")

    def reset_quotas(self):
        """Resets per-video quotas (Gemini)."""
        try:
            gemini_enhance = ImportGate.get("gemini_enhance")
            if gemini_enhance:
                gemini_enhance.quota_manager.reset()
        except Exception:
            pass
        except Exception as e:
            logger.warning(f"Failed to reset quotas: {e}")

    def process_video(self, video_path: str, aggressive: bool = False, keywords: str = None, retry_level: int = 0) -> str:
        """
        Main entry point. Gemini-Only Authority Mode.
        """
        # 🔒 SYSTEM LOCK ASSERTION
        watermark_source = "gemini"
        
        logger.info(f"🎬 Processing (Gemini Authority): {video_path} (Retry Level: {retry_level})")
        
        # INJECT FEEDBACK PROMPT FOR RETRIES
        if retry_level > 0:
             feedback_prompt = " FEEDBACK:PREVIOUS_FAILED_SEEK_DEEPER_AND_SMALLER MODE:NUCLEAR_SENSITIVITY "
             if keywords: keywords += feedback_prompt
             else: keywords = feedback_prompt
             logger.info(f"🧠 Injecting Feedback Prompt: {feedback_prompt.strip()}")
        
        if not os.path.exists(video_path):
             return self._error_json("Video file not found.")

        # 1. Select Representative Frame
        # Strategy: Try 20% mark. If empty/black, try 50%.
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if frame_count < 1:
            cap.release()
            return self._error_json("Empty video.")

        # 1. Select Representative Frames (Multi-Shot Strategy)
        # Scan 5%, 20%, 50%, 80%, 95% to ensure moving watermarks (slow reveal) are caught.
        frames_to_check = []
        scan_percentages = [0.05, 0.2, 0.5, 0.8, 0.95]
        
        for pct in scan_percentages:
            target_idx = int(frame_count * pct)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, scanned_frame = cap.read()
            if ret and scanned_frame is not None and np.mean(scanned_frame) > 10: # not black
                frames_to_check.append(scanned_frame)
        
        cap.release()
        
        if not frames_to_check:
             return self._error_json("Could not read any valid video frames.")
        
        # Use Middle Frame as reference for geometry and safety checks
        reference_frame = frames_to_check[len(frames_to_check)//2]
        
        job_start_time = time.time()
        
        # 2. Multi-Frame Detection
        detected_boxes = []
        try:
            gemini_enhance = ImportGate.get("gemini_enhance")
            # The ONE AND ONLY call (Strict Limit)
            call_num = 1
            logger.info(f"🔭 Initiating Gemini Detection (Call {call_num}/1) on {len(frames_to_check)} frames...")
            
            # STRICT CACHE POLICY: Inject Geometry Mode Salt
            # "Content Hash Policy Fix - NEVER reuse... If ANY differ -> FORCE fresh processing."
            salt = "MODE:GEOMETRY_LOCKED_STRICT"
            if keywords:
                 keywords = f"{keywords} | {salt}"
            else:
                 keywords = salt
            
            if keywords:
                 logger.info(f"    └─ 🔑 Injection: '{keywords[:50]}...'")
            
            detected_boxes = gemini_enhance.detect_watermark(frames_to_check, keywords=keywords)
            
            # 3a. FAIL-SAFE: Quota or Error -> Try FALLBACK
            if detected_boxes is None:
                logger.warning("⚠️ Gemini returned None (Quota/Error). Skipping (STRICT MODE).")
                return json.dumps({
                    "watermarks": [],
                    "count": 0,
                    "status": "CLEAN", 
                    "context": {"removal_success": False, "reason": "Gemini Quota/Error"}
                }, indent=2)
                
            # 3b. CLEAN STATE
            if not detected_boxes:
                logger.info("✅ Gemini reports NO watermarks found.")
                return json.dumps({
                    "watermarks": [],
                    "count": 0,
                    "status": "CLEAN", 
                    "context": {"removal_success": True}
                }, indent=2)
                
        except Exception as e:
            logger.error(f"❌ Gemini Detection Exception: {e}")
            # Exception in detection -> Safe Fail -> CLEAN
            return json.dumps({
                    "watermarks": [],
                    "count": 0,
                    "status": "CLEAN", 
                    "context": {"removal_success": False, "reason": f"Exception: {e}"}
            }, indent=2)

        # 3. Process & Validate Detected Boxes
        final_watermarks = []
        
        # Define Center Region (Normalized 0-1)
        # We reject boxes that intersect the center 40% (0.3 to 0.7)
        center_x_min = 0.3 * w_img
        center_x_max = 0.7 * w_img
        center_y_min = 0.3 * h_img
        center_y_max = 0.7 * h_img
        
        for i, box in enumerate(detected_boxes):
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            
            # --- VALIDATION 1: GEOMETRY ---
            # Aspect Ratio
            if h <= 0: continue
            ar = w / float(h)
            if ar < 0.05 or ar > 20.0:
                logger.warning("⚠️ [WM] Candidate rejected")
                logger.warning(f"    ├─ reason: aspect_ratio {ar:.2f}")
                logger.warning(f"    └─ box_id: {i}")
                continue
                
            # Area (Max 15%)
            area_pct = (w * h) / (w_img * h_img)
            if area_pct > 0.15:
                logger.warning("⚠️ [WM] Candidate rejected")
                logger.warning(f"    ├─ reason: area_too_large {area_pct:.1%}")
                logger.warning(f"    └─ box_id: {i}")
                continue
            if area_pct < 0.0001: # 0.01% Min
                # NUCLEAR MODE: Catch tiny text
                if retry_level > 0:
                     logger.info(f"☢️ Nuclear Mode: Accepting tiny watermark ({area_pct:.5f})")
                else: 
                     logger.info(f"Ignoring microscopic detection {i} (<0.01%)")
                     continue
                
            # Center Intersection Check REMOVED per user request
            # Relying solely on FaceProtector for safety.
            # -------------------------------------------------

            # --- VALIDATION 2: FACE SAFETY ---
            is_safe, reason = FaceProtector.is_safe_region(reference_frame, box)
            is_soft_warn = "SOFT_FACE_PROXIMITY" in reason
            
            if not is_safe:
                 logger.warning(f"⛔ Safety Reject (Face): {reason}")
                 continue
            
            if is_soft_warn:
                 logger.info(f"⚠️ [WM] Allowed with Soft Face Proximity warning.")
                 
            # --- CONFIDENCE CALCULATION ---
            # Base confidence
            conf = 0.95
            
            # Penalize if near edges? No, watermarks ARE near edges.
            # Penalize if "floating" in middle-ish (but passed center check)
            # Actually, standard confidence logic:
            # If it passed all hard gates, we are fairly confident.
            # Let's adjust based on Area/AR sanity? 
            if area_pct > 0.10: conf -= 0.1 # Large-ish
            if ar < 0.2 or ar > 5.0: conf -= 0.1 # Weird shape
            
            # Final confidence clamp
            conf = max(0.0, min(1.0, conf))
            
            # Safety Gate
            # NUCLEAR MODE: Lower confidence floor significantly
            min_conf = 0.45 if retry_level > 0 else 0.7
            
            if conf < min_conf:
                 logger.warning("⚠️ [WM] Candidate rejected")
                 logger.warning(f"    ├─ reason: low_confidence {conf:.2f} (Threshold: {min_conf})")
                 logger.warning(f"    └─ box_id: {i}")
                 continue
            
            if retry_level > 0: logger.info(f"☢️ Nuclear Mode: Accepted low confidence ({conf:.2f}) candidate.")
            
            # --- VERIFICATION STEP ---
            # Double-check purely clean videos to prevent hallucinations.
            if retry_level == 0:
                try:
                    g_verify = ImportGate.get("gemini_enhance")
                    if g_verify:
                        # Verify the crop
                        is_confirmed = g_verify.verify_watermark(reference_frame, box)
                        if is_confirmed is False:
                            logger.info(f"❌ Verification Reject: Gemini flagged candidate {i} as false positive.")
                            continue
                        elif is_confirmed is True:
                            logger.info(f"✅ Verification Pass: Gemini confirmed candidate {i}.")
                except Exception as vx:
                    logger.warning(f"Verification warning: {vx}")
                 
            # 🛡️ RULE 2: ABSOLUTE ACCURACY GEOMETRY (Aspect-Aware V2)
            # 1. Normalize Ratio
            ar = w / float(h)
            
            if ar > 3.0: 
                # Wide (Text line?) -> Expand Height significantly types
                exp_w_pct = 0.25 # +25% Width
                exp_h_pct = 0.45 # +45% Height
            elif ar < (1.0/2.5): # h/w > 2.5
                # Tall (Vertical banner/Logo?) 
                exp_w_pct = 0.40 # +40% Width
                exp_h_pct = 0.40 # +40% Height
            else:
                # Standard (Square-ish Logo)
                exp_w_pct = 0.30 # +30% Width
                exp_h_pct = 0.30 # +30% Height
                
            exp_w = int(w * exp_w_pct)
            exp_h = int(h * exp_h_pct)
            
            # Apply Initial Centered Expansion
            nx = x - (exp_w // 2)
            ny = y - (exp_h // 2)
            nw = w + exp_w
            nh = h + exp_h
            
            # Clamp 1
            nx = max(0, nx); ny = max(0, ny)
            nw = min(nw, w_img - nx); nh = min(nh, h_img - ny)
            
            # 🛡️ 2. EDGE COVERAGE VALIDATION (Canny Check)
            # Only if we have the reference frame (we do)
            try:
                # Extract Candidate
                check_roi = reference_frame[ny:ny+nh, nx:nx+nw]
                if check_roi.size > 0:
                    gray_roi = cv2.cvtColor(check_roi, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(gray_roi, 60, 120)
                    
                    # Check Borders for Edges
                    # Top, Bottom, Left, Right
                    # If > 5% of border pixels are edges -> Expand
                    
                    pad_add_x = 0
                    pad_add_y = 0
                    extra_pad = 8 # +8px (Strict)
                    
                    # Top
                    if np.count_nonzero(edges[0,:]) > (nw * 0.05):
                        ny = max(0, ny - extra_pad)
                        nh += extra_pad
                        
                    # Bottom
                    if np.count_nonzero(edges[-1,:]) > (nw * 0.05):
                        nh += extra_pad
                        
                    # Left
                    if np.count_nonzero(edges[:,0]) > (nh * 0.05):
                        nx = max(0, nx - extra_pad)
                        nw += extra_pad
                        
                    # Right
                    if np.count_nonzero(edges[:,-1]) > (nh * 0.05):
                        nw += extra_pad
                        
                    # Re-Clamp
                    nx = max(0, nx); ny = max(0, ny)
                    nw = min(nw, w_img - nx); nh = min(nh, h_img - ny)
            except Exception as e:
                logger.warning(f"Edge Validation failed: {e}")
            
            # Update Box (Final Lock)
            box['x'], box['y'], box['w'], box['h'] = nx, ny, nw, nh
            
            watermark_entry = {
                "id": i+1,
                "coordinates": box,
                "confidence": conf,
                "safe_to_remove": True,
                "decision": "remove",
                "time_range": {"start": 0.0, "end": 0.0}, # Static assumption
                "is_moving": box.get("motion_hint") == "dynamic",
                "motion_hint": box.get("motion_hint", "static"),
                "watermark_type": "GEMINI_EXACT",
                "face_proximity": is_soft_warn,
                "strategy": "inpaint_standard", # Default
                "semantic_class": box.get("semantic_type", "unknown")
            }
            final_watermarks.append(watermark_entry)
            
        # 4. Final Packaging
        # Only save frame if we actually found something to remove
        if final_watermarks:
            frame_path = os.path.join(os.path.dirname(video_path), f"frame_{uuid.uuid4().hex[:6]}.jpg")
            try:
                cv2.imwrite(frame_path, reference_frame)
            except: 
                frame_path = "error_saving_frame.jpg"
                
            logger.info(f"🏁 Finalized {len(final_watermarks)} watermarks for removal.")

            return json.dumps({
                "watermarks": final_watermarks,
                "count": len(final_watermarks),
                "status": "DETECTED",
                "context": {"frame_path": frame_path, "removal_success": False},
                "processing_time": time.time() - job_start_time
            }, indent=2)
        else:
            # Clean
            return json.dumps({
                "watermarks": [],
                "count": 0,
                "status": "CLEAN",
                "context": {"removal_success": True}
            }, indent=2)

    def generate_static_mask(self, video_path: str, box: dict, output_path: str, padding_ratio: float = 0.0, semantic_class: str = "text") -> bool:
        """
        Generates a STATIC mask video using Alpha-Safe Authority logic.
        """
        try:
            # Save original box for Geometry Lock
            original_box = box.copy()
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return False
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
            
            # --- TEMPORAL SMART REFINEMENT (Production-Grade) ---
            # Now uses N=5 Frames + Consensus Voting to eliminate drift/blobs.
            try:
                from opencv_watermark import TemporalSmartRefiner
                
                refined_box = TemporalSmartRefiner.refine_box_temporal(video_path, box)
                box = refined_box # Update box to be the tight fit (includes micro-pad)
                
                # DISABLE downstream padding logic because TemporalRefiner already micro-padded
                # We want 0 additional padding.
                force_zero_padding = True
                
            except Exception as e:
                logger.warning(f"Temporal Refinement skipped: {e}")
                force_zero_padding = False
            
            # --- 1. GEOMETRY LOCK (Alpha-Safe) ---
            # Geometry Lock has been removed/deprecated
            # box = original_box.copy()

            x, y, w, h = box['x'], box['y'], box['w'], box['h']

            # --- 2. ALPHA-SAFE MASK GENERATION ---
            # Create Core Mask (Pixel-Perfect from Locked Box)
            core_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(core_mask, (x, y), (x + w, y + h), 255, -1)
            

            
            # Smart Halo (Alpha Fringe)
            # We need a frame for gradient check. Use middle frame?
            # We already released 'cap'.
            # Re-open for one frame? Or use a blank? Halo needs image content.
            # Efficiency: Just open once.
            
            final_mask = core_mask
            
            try:
                cap_ref = cv2.VideoCapture(video_path)
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret_ref, frame_ref = cap_ref.read()
                cap_ref.release()
                
                if ret_ref:
                    # Halo logic deprecated/removed
                    pass
                    
                    # final_mask defaults to core_mask
                    final_mask = core_mask
                    
                    # --- PART C: Glyph-Safe Expansion (TEXT ONLY) ---
                    # Determinstic expansion to fix thin horizontal strokes
                    # RULE 3: GLYPH EXPANSION GUARD (Skip if CPU Safe Dynamic Feather active)
                    is_cpu_safe = os.getenv("COMPUTE_MODE") == "cpu"
                    
                    if semantic_class == "text" and not is_cpu_safe:
                        # Expand by: +2px Vertically (Radius 1), +3px Horizontally (Radius ~1-2)
                        # We use Kernel Size: (Width, Height)
                        # Height: +2px total -> Kernel 3 (1px up, 1px down) ? Or Radius 2? 
                        # User said "Expand MASK BY...". Usually implies radius.
                        # Let's use Kernel (7, 5) -> Rad 3 (H), Rad 2 (V). 
                        # Reason: "Anti-aliasing leaks outside".
                        
                        logger.info(f"✍️ Applying Glyph-Safe Expansion (+2px V, +3px H) for {semantic_class}...")
                        kernel_glyph = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)) # W=7, H=5
                        expanded_mask = cv2.dilate(final_mask, kernel_glyph, iterations=1)
                        
                        # Constraint: ONLY inside origin_box (Strict)
                        constraint_canvas = np.zeros_like(final_mask)
                        ox, oy, ow, oh = original_box['x'], original_box['y'], original_box['w'], original_box['h']
                        # Ensure coords inside frame
                        ox = max(0, ox); oy = max(0, oy)
                        ow = min(ow, width - ox); oh = min(oh, height - oy)
                        
                        cv2.rectangle(constraint_canvas, (ox, oy), (ox+ow, oy+oh), 255, -1)
                        final_mask = cv2.bitwise_and(expanded_mask, constraint_canvas)

            except Exception as e:
                logger.warning(f"Smart Halo generation failed: {e}")
                final_mask = core_mask

            # --- 3. FINAL SAFETY CLIP (Face Firewall) ---
            # Ensure we NEVER touch the face, even if mask expanded or drifted.
            try:
                # Use Reference Frame for detection
                cap_safety = cv2.VideoCapture(video_path)
                cap_safety.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
                ret_s, frame_s = cap_safety.read()
                cap_safety.release()
                
                if ret_s:
                    faces = FaceProtector.detect_faces(frame_s)
                    for (fx, fy, fw, fh) in faces:
                         # DEFINE CORE ZONE (Top 85%)
                         # User Policy: Face = Protected, Neck = Allowed.
                         protected_h = int(fh * 0.85)

                         # CLIP MASK (Set to 0)
                         # We use -1 thickness to fill
                         cv2.rectangle(final_mask, (fx, fy), (fx+fw, fy+protected_h), 0, -1)
                         # Clarified log: It's a preventive measure, not necessarily a collision.
                         logger.info(f"🛡️ Face Firewall: Sanitized Face Region at ({fx},{fy}) - Top 85% Protected")
                        
                         # Bottom 15% (Neck) is left UNTOUCHED.
            except Exception as e:
                logger.warning(f"Face Firewall failed: {e}")

            mask_frame = final_mask
            
            # Write Video (GRAYSCALE / MONO)
            # FFmpeg mask inputs usually expect YUV, so we write a standard grayscale video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # isColor=False for 1-channel output
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
            
            if not out.isOpened():
                logger.error("Failed to open video writer for mask.")
                return False

            # Optimization: Write chunk? No, write full duration as requested.
            # To speed up, we could loop, but for stability we just loop count.
            for _ in range(total_frames):
                 out.write(mask_frame)
                 
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Mask Gen Error: {e}")
            return False

    def generate_tracked_mask(self, video_path: str, box: dict, output_path: str, padding_ratio: float = 0.0, semantic_class: str = "text") -> bool:
        """
        Smart Tracking Mask Generator (CPU-Safe).
        Uses Template Matching with Local Search Window to follow moving watermarks.
        Non-destructive addition.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return False
            
            w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 1. Extract Template from Reference Frame (Middle Frame usually reliable)
            # Or assume 'box' corresponds to the middle frame used in detection?
            # Detection uses frames_to_check list.
            # Let's verify 'box' provenance. Usually it's from the reference_frame (middle).
            
            # We'll scan a few frames to find a good template if needed, 
            # but for now assume box is valid for the middle frame.
             
            ref_idx = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            ret, ref_frame = cap.read()
            
            if not ret: 
                cap.release()
                return False
                
            x, y, w, h = box['x'], box['y'], box['w'], box['h']
            
            # Safety Clamp
            x = max(0, x); y = max(0, y)
            w = min(w, w_vid - x); h = min(h, h_vid - y)
            
            if w < 10 or h < 10:
                cap.release()
                return False # Too small to track
                
            template = ref_frame[y:y+h, x:x+w]
            if template.size == 0:
                cap.release()
                return False
                
            # Initialize Tracker State
            curr_x, curr_y = x, y
            search_margin = max(50, int(max(w, h) * 0.5)) # Search window size
            
            # Prepare Output
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w_vid, h_vid), isColor=False)
            
            # Optimization: Pre-calculate padding kernel if needed
            kernel_glyph = None
            if semantic_class == "text":
                kernel_glyph = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
            
            # Rewind
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Loop
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret: break
                
                # --- SMART TRACKING ---
                # Define Search Window around LAST known position (curr_x, curr_y)
                sx = max(0, curr_x - search_margin)
                sy = max(0, curr_y - search_margin)
                ex = min(w_vid, curr_x + w + search_margin)
                ey = min(h_vid, curr_y + h + search_margin)
                
                # Extract Search Region
                search_region = frame[sy:ey, sx:ex]
                
                found_new_pos = False
                
                if search_region.shape[0] > h and search_region.shape[1] > w:
                    try:
                        # Template Match
                        res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > 0.65: # Threshold: Reasonable match
                            # Update Position (Global Coords)
                            # max_loc is relative to search_region (sy, sx)
                            new_x = sx + max_loc[0]
                            new_y = sy + max_loc[1]
                            
                            # Smooth update (Weighted average? Or instant?)
                            # Watermarks usually move rigidly. Instant is better to avoid lag.
                            curr_x, curr_y = new_x, new_y
                            found_new_pos = True
                    except: pass
                
                # If match failed (occlusion/off-screen), we keep 'curr_x, curr_y' (Last Known Pos)
                
                # --- DRAW MASK ---
                mask_frame = np.zeros((h_vid, w_vid), dtype=np.uint8)
                
                # Apply Dynamic Padding
                # Add 'padding_ratio'
                px = int(w * padding_ratio)
                py = int(h * padding_ratio)
                
                draw_x = curr_x - px
                draw_y = curr_y - py
                draw_w = w + (px * 2)
                draw_h = h + (py * 2)
                
                # Draw Core
                cv2.rectangle(mask_frame, (draw_x, draw_y), (draw_x+draw_w, draw_y+draw_h), 255, -1)
                
                # Apply Glyph Exp (if text)
                if kernel_glyph is not None:
                     # Limit dilation to the ROI to save speed
                     # Actually on full frame is fast enough for binary mask
                     mask_frame = cv2.dilate(mask_frame, kernel_glyph, iterations=1)
                     
                # --- FACE SAFETY (Per Frame) ---
                # We can reuse the static method if we import FaceProtector
                # But we are inside HybridWatermark which imports ...
                # Wait, hybrid_watermark imports FaceProtector at top level.
                
                # Optimization: Only run face check every N frames? 
                # No, faces move. Run every frame but efficiently?
                # Actually, FaceProtector.detect_faces uses HumanGuard (DNN) which is slow on CPU?
                # If CPU mode, it might be slow.
                # But user wanted "Smart".
                # Let's do a simple check: Check if mask intersects "Center Top" if we want to skip detection?
                # No, FaceProtector.is_safe_region logic uses FaceProtector.detect_faces.
                # Let's perform the clip.
                try:
                    faces = FaceProtector.detect_faces(frame)
                    for (fx, fy, fw, fh) in faces:
                         # Core Face Zone (Top 85%)
                         core_h = int(fh * 0.85)
                         # punch hole
                         cv2.rectangle(mask_frame, (fx, fy), (fx+fw, fy+core_h), 0, -1)
                except: pass

                out.write(mask_frame)
                
            cap.release()
            out.release()
            logger.info(f"📍 Tracked Mask Generated: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Tracked Mask Gen Error: {e}")
            return False

# Singleton
hybrid_detector = HybridWatermarkDetector()
