
import cv2
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

class StaticPatchReuseEngine:
    """
    Engine for accelerating inpainting by reusing a high-quality static patch 
    across frames, using motion tracking (Rigid) or static placement.
    """
    
    @staticmethod
    def analyze_stability(video_path, mask_paths, motion_hint="unknown", video_duration=0.0):
        """
        Determines the optimal removal strategy.
        """
        # Trust the hint from Gemini if available
        if motion_hint in ["static", "rigid_motion", "dynamic"]:
            return motion_hint
            
        # Fallback Heuristics
        # If video is short (< 5s), bias towards Rigid Motion (camera shake unlikely to shift perspective much)
        if video_duration > 0 and video_duration < 5.0:
            return "rigid_motion"
            
        return "dynamic" # Default fail-safe

    @staticmethod
    def check_pixel_motion(video_path, box, threshold=5.0):
        """
        Detects if the pixels INSIDE the watermark box are moving (Camera shake vs Tripod).
        Returns True if motion detected.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return True # Assume dynamic on error
            
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample 5 frames
            indices = np.linspace(0, total-5, 5).astype(int)
            
            prev_gray = None
            max_diff = 0.0
            
            x,y,w,h = box['x'], box['y'], box['w'], box['h']
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret: continue
                
                # Crop
                h_img, w_img = frame.shape[:2]
                cx = max(0, x); cy = max(0, y)
                cw = min(w, w_img - cx); ch = min(h, h_img - cy)
                
                if cw <= 0 or ch <= 0: continue
                crop = frame[cy:cy+ch, cx:cx+cw]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                
                if prev_gray is not None:
                     if prev_gray.shape != gray.shape: continue
                     
                     diff = cv2.absdiff(prev_gray, gray)
                     score = np.mean(diff)
                     if score > max_diff: max_diff = score
                
                prev_gray = gray
                
            cap.release()
            
            is_moving = max_diff > threshold
            logger.info(f"🎥 Watermark Motion Scan: Score={max_diff:.2f} (Threshold={threshold}) -> Mov:{is_moving}")
            return is_moving
            
        except Exception as e:
            logger.warning(f"Motion check failed: {e}")
            return True # Fallback to dynamic

    @staticmethod
    def apply_patch(video_path, mask_paths, output_path, mode="static"):
        """
        Applies a static or rigid-motion tracked patch with PADDED CONTEXT and OUTWARD FADE.
        """
        if not mask_paths: return False
        
        mask_path = mask_paths[0]
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): return False
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Get Mask ROI
            cap_m = cv2.VideoCapture(mask_path)
            ret_m, mask_frame = cap_m.read()
            cap_m.release()
            
            if not ret_m: 
                cap.release()
                return False
                
            if len(mask_frame.shape) == 3:
                mask_gray = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_frame
                
            x, y, bw, bh = cv2.boundingRect(mask_gray)
            if bw < 10 or bh < 10: 
                cap.release()
                return False
            
            # --- PADDING STRATEGY (Buffer Zone) ---
            pad = 8 # Tight fit to avoid ghosting on nearby objects
            px1 = max(0, x - pad)
            py1 = max(0, y - pad)
            px2 = min(w, x + bw + pad)
            py2 = min(h, y + bh + pad)
            
            # Width/Height of the PADDED patch
            pw = px2 - px1
            ph = py2 - py1
            
            # The mask relative to the padded patch
            rel_x = x - px1
            rel_y = y - py1
            
            # Extract Padded Mask
            # We need the mask to cover the watermark, but the inpainting needs context.
            # We will use the original mask for inpainting guidance.
            patch_mask_full = mask_gray[py1:py2, px1:px2]
            
            # 2. Extract Template & Create Base Patch
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) // 2))
            ret, frame = cap.read()
            if not ret: 
                cap.release()
                return False
                
            roi_padded = frame[py1:py2, px1:px2]
            
            # HYBRID INPAINTING STRATEGY (Restored)
            # Combine Navier-Stokes (Structure) and Telea (Smoothness).
            
            # 1. Generate Navier-Stokes (Structure / "DNA")
            patch_ns = cv2.inpaint(roi_padded, patch_mask_full, 3, cv2.INPAINT_NS)
            
            # 2. Generate Telea (Smoothness / "Lens")
            patch_telea = cv2.inpaint(roi_padded, patch_mask_full, 3, cv2.INPAINT_TELEA)
            
            # 3. Blend them (50/50 Hybrid)
            patch_clean = cv2.addWeighted(patch_ns, 0.5, patch_telea, 0.5, 0)
            
            # --- OUTWARD FADE BLENDING SETUP ---
            # 1. Start with the binary watermark mask (0 or 255)
            # 2. Dilate it to ensure we cover the edges (Safe Zone)
            # 3. Blur it to create the fade OUT into the buffer zone (Reserved for fallback)
            
            # Create float alpha
            alpha_base = np.zeros((ph, pw), dtype=np.uint8)
            alpha_base = patch_mask_full.copy()
            
            # Dilate to create solid core (Safe Zone)
            # INCREASED to (9,9) to fix "Shape/Location" issues (Clipping)
            # We must cover the text fully for seamlessClone to work.
            kernel_core = np.ones((9, 9), np.uint8) 
            alpha_core = cv2.dilate(alpha_base, kernel_core, iterations=1)
            
            # Gaussian Blur for the Fade-Out (Softer Edge)
            alpha_blur = cv2.GaussianBlur(alpha_core, (7, 7), 0) 
            
            # Normalize to 0..1 (Useful for manual blend fallback if seamlessClone fails)
            patch_alpha = alpha_blur.astype(float) / 255.0
            patch_alpha_3c = cv2.merge([patch_alpha, patch_alpha, patch_alpha])

            
            # 4. Tracking & Rendering Loop
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

            # TRACKING STATE (Using original coordinates for tracking logic logic, but rendering padded)
            # Padded offset from tracked point:
            # The tracker tracks (x,y). We render at (x-pad_l, y-pad_t).
            # But wait, logic below tracks top-left of the bounding box.
            
            paste_x, paste_y = x, y # Track the WATERMARK pos
            
            # Offsets for rendering
            off_x = x - px1 # How much "left" we go from the tracked point
            off_y = y - py1
            
            smooth_x, smooth_y = None, None
            smooth_diff = None
            
            wm_template = cv2.cvtColor(roi_padded[rel_y:rel_y+bh, rel_x:rel_x+bw], cv2.COLOR_BGR2GRAY)

            do_track = (mode == "rigid_motion")
            
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                curr_h, curr_w = frame.shape[:2]
                
                # --- STABILIZED TRACKING ---
                if do_track:
                    s_margin = 60
                    exp_x, exp_y = paste_x, paste_y
                    
                    sx1 = int(max(0, exp_x - s_margin))
                    sy1 = int(max(0, exp_y - s_margin))
                    sx2 = int(min(curr_w, exp_x + bw + s_margin))
                    sy2 = int(min(curr_h, exp_y + bh + s_margin))
                    
                    search_roi = frame[sy1:sy2, sx1:sx2]
                    
                    if search_roi.shape[0] > bh and search_roi.shape[1] > bw:
                        gray_roi = cv2.cvtColor(search_roi, cv2.COLOR_BGR2GRAY)
                        res = cv2.matchTemplate(gray_roi, wm_template, cv2.TM_CCOEFF_NORMED)
                        _, max_val, _, max_loc = cv2.minMaxLoc(res)
                        
                        if max_val > 0.4:
                             target_x = sx1 + max_loc[0]
                             target_y = sy1 + max_loc[1]
                             
                             if smooth_x is None:
                                 smooth_x, smooth_y = float(target_x), float(target_y)
                             else:
                                 smooth_x = smooth_x * 0.5 + target_x * 0.5
                                 smooth_y = smooth_y * 0.5 + target_y * 0.5
                                 
                             paste_x, paste_y = int(smooth_x), int(smooth_y)
                
                # --- RENDER PADDED PATCH ---
                # Calculate Top-Left of the PAD box
                render_x = int(paste_x) - off_x
                render_y = int(paste_y) - off_y
                
                if render_x+pw <= curr_w and render_y+ph <= curr_h and render_x >= 0 and render_y >= 0:
                     roi_target = frame[render_y:render_y+ph, render_x:render_x+pw]
                     
                     # POISSON BLENDING (Seamless Clone)
                     # This calculates gradients to match the destination context perfectly, removing the "scar".
                     # We use the tight mask (alpha_core) to define the "DNA" shape.
                     
                     center = (pw // 2, ph // 2)
                     
                     try:
                         # normalize patch_clean to uint8 if not
                         p_src = patch_clean.astype(np.uint8)
                         p_dst = roi_target.astype(np.uint8)
                         p_mask = alpha_core # The tight 3px dilated mask
                         
                         # NORMAL_CLONE: Replaces texture (Good for hiding, eliminates ghosting)
                         # MIXED_CLONE: Failed (Text bleeds through).
                         # Reverting to NORMAL_CLONE as user priority is REMOVAL, not blending.
                         
                         clone = cv2.seamlessClone(p_src, p_dst, p_mask, center, cv2.NORMAL_CLONE)
                         roi_target[:] = clone
                     except Exception as e:
                         # Fallback to simple copy if SeamlessClone fails (e.g. dimensions)
                         # But let's log? No logger here. Just fallback safe.
                         logger.warning(f"SeamlessClone failed: {e}")

                out.write(frame)
                
            cap.release()
            out.release()
            return True
            
        except Exception as e:
            logger.error(f"Static Patch Engine Fault: {e}")
            return False
