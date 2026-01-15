"""
Automated Watermark Orchestrator
Centralizes adaptive removal logic to unburden compiler.py.
"""

import os
import shutil
import logging
from typing import List, Dict, Tuple, Optional

# Shared Modules
import hybrid_watermark
# Shared Modules
from import_gate import ImportGate

from opencv_watermark import inpaint_video, check_watermark_residue, MicroTextureBlender, MaskVerifier, verify_visual_guarantee

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("watermark_auto")

def run_adaptive_watermark_orchestration(
    input_video: str, 
    watermarks: List[Dict], 
    output_video: str, 
    job_dir: str, 
    original_height: int = 1080, 
    aggressive_mode: bool = True, # Kept for API compatibility
    retry_level: int = 0
) -> Tuple[bool, str]:
    """
    Control System for Watermark Removal.
    Iteratively tunes mask padding and inpaint radius based on residue feedback.
    Moved from compiler.py to reduce bloat.
    """
    logger.info(f"🧠 Starting Smart-Decay Watermark Orchestrator (Retry Level: {retry_level})...")
    
    # LEVEL-BASED STRATEGY OVERRIDES
    # Level 0: Standard Auto
    # Level 1: Aggressive Static (Radius +3)
    # Level 2: Better Accurate Patch (Radius +6)
    
    # 1. Check Motion First (ALWAYS)
    # Get the box from the first watermark (primary)
    wm_box = watermarks[0]['coordinates'] if watermarks else {'x':0,'y':0,'w':0,'h':0}
    
    from static_patch_engine import StaticPatchReuseEngine
    is_moving = StaticPatchReuseEngine.check_pixel_motion(input_video, wm_box)
    
    motion_override = "static"
    radius_boost_level = 0
    
    if is_moving:
         logger.info("🌊 Motion Detected: Forcing DYNAMIC Mode (Frame-by-Frame).")
         motion_override = "dynamic"
         # Dynamic needs slight boost to blend
         radius_boost_level = 2
    else:
         if retry_level >= 2:
             logger.info("🎯 Retry Level 2: Target 'Better Accurate Patch' (Static +6px).")
             radius_boost_level = 6
             motion_override = "static"
         elif retry_level == 1:
             logger.info("🔥 Retry Level 1: Aggressive Static (Static +3px).")
             radius_boost_level = 3
             motion_override = "static"
         else:
             logger.info("🧱 Static Watermark Detected (Standard Auto).")
             radius_boost_level = 0
             motion_override = "static"

    # SMART-DECAY STRATEGY DEFINITIONS
    strategies = [
        {'pad': 0.15, 'rad': 5, 'thresh': 0.15, 'name': 'Precision (Strict)'},
        {'pad': 0.25, 'rad': 7, 'thresh': 0.40, 'name': 'Balanced (Medium)'},
        {'pad': 0.35, 'rad': 9, 'thresh': 1.00, 'name': 'Nuclear (Force)'}
    ]
    
    # If original video is low res, boost radius slightly for all
    radius_boost = (1 if original_height < 1080 else 0) + radius_boost_level
    
    
    for attempt, strat in enumerate(strategies, 1):
        pad_ratio = strat['pad']
        # Apply Logic Boost
        radius = strat['rad'] + radius_boost
        threshold = strat['thresh']
        name = strat['name']
        
        logger.info(f"🛡️ Strategy {attempt}/{len(strategies)}: [{name}] (Motion: {motion_override})")
        logger.info(f"   ├─ P:{pad_ratio:.2f} | R:{radius}px | Limit:{threshold:.2f}")
        
        # 1. Generate Masks
        masks = []
        text_masks = []
        for i, wm in enumerate(watermarks):
            mpath = os.path.join(job_dir, f"mask_a{attempt}_{i}.mp4")
            
            # Use Standard Static Mask (User Requested: Static Cleaning Only)
            gen_success = hybrid_watermark.hybrid_detector.generate_static_mask(
                input_video, 
                wm['coordinates'], 
                mpath, 
                padding_ratio=pad_ratio, 
                semantic_class=wm.get("semantic_class", "unknown")
            )

            if gen_success:
                masks.append(mpath)
                if wm.get("semantic_class") == "text":
                    text_masks.append(mpath)
        
        if not masks: return False, "Mask Gen Failed"
        
        # Safe Iteration
        worst_uncovered_ratio = 0.0
        
        for mpath in masks:
            try:
                # Parse index from filename "mask_aX_Y.mp4"
                basename = os.path.basename(mpath)
                parts = basename.split('_')
                wm_idx = int(parts[-1].split('.')[0])
                if wm_idx < len(watermarks):
                    wm_box = watermarks[wm_idx]['coordinates']
                    
                    # Verify & Fix
                    uncovered = MaskVerifier.check_and_fix_coverage(input_video, mpath, wm_box)
                    if uncovered > worst_uncovered_ratio:
                        worst_uncovered_ratio = uncovered
            except Exception as e:
                logger.warning(f"Mask verification skipped for {mpath}: {e}")
        
        # 2. Inpaint
        out_candidate = os.path.join(job_dir, f"candidate_a{attempt}.mp4")
        success = inpaint_video(input_video, masks, out_candidate, original_height=original_height, radius_override=radius, motion_hint_override=motion_override)
        
        if not success:
            logger.warning("❌ Inpaint execution failed.")
            return False, "Inpaint Exec Fail"

        # 🔒 PART A: HARD STOP FOR STRATEGY LOOP
        # IF inpaint_executed == True: DO NOT advance strategy.
        # This converts system to manual retry.
        
        logger.info("🧱 Strategy Loop Disabled — Manual Control Active")
        
        # 3. Assess Residue (For Logging Only)
        wm_boxes = [wm['coordinates'] for wm in watermarks]
        res = check_watermark_residue(input_video, out_candidate, masks, watermark_boxes=wm_boxes)
        score = res.get("score", 1.0)
        reason = res.get("reason", "")
        
        logger.info(f"🧪 Residue Judge:")
        logger.info(f"   ├─ Score: {score:.2f} (Target <= {threshold:.2f})")
        logger.info(f"   └─ Reason: {reason}")
        
        # Micro-Texture Logic (Keep existing logic just for blend application)
        should_texture = True
        if worst_uncovered_ratio > 0.02:
             logger.warning(f"⚠️ MicroTexture skipped — origin coverage was incomplete ({worst_uncovered_ratio:.1%}).")
             should_texture = False
        
        if attempt == 1 and text_masks and should_texture:
             # DISABLED: User requests Clean/Diluted look, NOT grain.
             logger.info("🎨 Skipping Micro-Texture Blend (User Pref: Dilution/Blur)")
             # MicroTextureBlender.apply_texture_blend(out_candidate, text_masks, out_candidate)
        
        # 🔒 PART D: VISUAL GUARANTEE CHECK
        # "Before committing sample ROI... if mean_lum < T AND edge > T... mark as VISIBLE"
        is_visible, viz_reason = verify_visual_guarantee(input_video, out_candidate, masks)
        
        if is_visible:
            logger.warning(f"⚠️ Visible watermark detected — awaiting user decision ({viz_reason})")
            # Logic: Return success=True (so pipeline continues to User Review) but with Warning Context?
            # Or reliance on manual inspection.
            # The prompt says: "Even if residue is high, control returns to user to user NO."
        else:
            logger.info("✅ Visual Guarantee Passed (Clean).")

        shutil.move(out_candidate, output_video)
        
        # Break Loop - We only run Strategy 1
        # Decide Final Status based on Score
        status_msg = f"Completed (Score:{score:.2f})"
        final_success = True
        
        # AUTO RECOVERY Check (Only for Level 0)
        if score > 0.15 and retry_level == 0: 
             logger.warning(f"⚠️ High Residue Detected (Score {score:.2f}). Triggering ONE-SHOT Auto Recovery...")
             
             recover_path = os.path.join(job_dir, f"recovery_a{attempt}.mp4")
             
             # Re-check motion logic here locally just for the auto-recovery branch
             # If it was dynamic effectively, we might strictly need it
             # But we can just rely on the overrides we calculated at top? 
             # No, auto-recovery logic was specific:
             # "If moving -> dynamic. If static -> aggressive static."
             # Use the same 'is_moving' flag we calculated earlier.
             
             rec_mode = "dynamic" if is_moving else "static_aggressive"
             rec_rad = radius + 2 if is_moving else radius + 5
             rec_hint = "dynamic" if is_moving else "static"
             
             logger.info(f"🚑 Auto-Recovery Mode: {rec_mode}")
             
             recover_success = inpaint_video(
                 input_video, masks, recover_path, 
                 original_height=original_height, 
                 radius_override=rec_rad, 
                 motion_hint_override=rec_hint
             )
             
             if recover_success:
                 shutil.move(recover_path, output_video)
                 status_msg = f"AutoRecovered_{rec_mode}"

        if score > 0.5:
             status_msg += " (Review_Required)"
        
        return final_success, status_msg



# Backwards Compatibility / Legacy Redirect
# This ensures 'reprocess_watermark_step' in compiler.py works if it tries to use simpler logic,
# OR we can upgrade it to use the full pipeline.
def process_video_with_watermark(input_path: str, output_path: str, retry_mode: bool = False) -> Dict:
    """
    Legacy wrapper upgraded to use the advanced pipeline.
    Auto-detects watermarks and runs orchestration.
    """
    logger.info(f"🛡️ Legacy Process wrapper called for: {input_path}")
    
    # 1. Detect
    import hybrid_watermark
    # We need a job dir
    job_dir = os.path.join("temp_watermark", f"job_{int(os.path.getmtime(input_path))}")
    os.makedirs(job_dir, exist_ok=True)
    
    try:
        watermarks = hybrid_watermark.hybrid_detector.detect_watermarks(input_path)   
        if not watermarks:
             logger.info("No watermarks detected (legacy wrapper).")
             shutil.copy(input_path, output_path)
             return {"success": True, "context": None}
             
        # 2. Orchestrate
        success, reason = run_adaptive_watermark_orchestration(
            input_path, watermarks, output_path, job_dir, 
            aggressive_mode=retry_mode # Retry mode enables aggressive
        )
        
        return {"success": success, "context": {"reason": reason, "removal_success": success, "watermark_status": "DETECTED_AND_REMOVED" if success else "DETECTED_BUT_FAILED"}}
    except Exception as e:
        logger.error(f"Legacy wrapper failed: {e}")
        return {"success": False, "context": None}
    finally:
        # Cleanup temp job dir? Or leave for pruner?
        # Leave for pruner since it's temp_watermark
        pass

def apply_text_watermark(*args, **kwargs): return False
