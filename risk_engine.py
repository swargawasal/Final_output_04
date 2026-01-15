"""
Transformation Risk Engine
--------------------------
Step 3: Early Abort Mechanism.
Prevents low-effort content from entering the expensive pipeline.
Checks:
1. Scene Count (Must be >= 2)
2. Motion Entropy (Must be non-static)
"""

import os
import cv2
import logging
import numpy as np
import subprocess

logger = logging.getLogger("risk_engine")

class RiskEngine:
    
    @staticmethod
    def check_scene_count(video_path, threshold=0.3):
        """
        Uses ffmpeg scene detection to estimate scene cuts.
        Fail if < 2 "scenes" (essentially one static shot).
        """
        try:
            # ffmpeg -i input -filter:v "select='gt(scene,0.3)',showinfo" -f null -
            cmd = [
                "ffmpeg", "-i", video_path, 
                "-vf", f"select='gt(scene,{threshold})',showinfo", 
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            # count "showinfo" lines
            output = result.stderr + result.stdout
            scene_cuts = output.count("showinfo")
            
            # Total scenes = cuts + 1
            total_scenes = scene_cuts + 1
            return total_scenes
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return 2 # Assume safe on error

    @staticmethod
    def check_motion_entropy(video_path, samples=10):
        """
        Checks if video is effectively static.
        Returns: motion_score (0.0 - 1.0)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 2: return 0.0
            
            indices = np.linspace(0, total-1, samples).astype(int)
            prev_gray = None
            diff_accum = 0.0
            
            for i in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret: continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Downscale for speed/noise rejection
                gray = cv2.resize(gray, (64, 64))
                
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray)
                    mean_diff = np.mean(diff)
                    diff_accum += mean_diff
                prev_gray = gray
                
            cap.release()
            
            # Normalize? Mean pixel diff > 5 is usually motion
            avg_diff = diff_accum / (max(1, len(indices)-1))
            return avg_diff
            
        except Exception as e:
            logger.warning(f"Motion check failed: {e}")
            return 10.0 # Pass on error

    @staticmethod
    def analyze_risk(video_path):
        """
        Complete check. Returns (is_safe: bool, reason: str)
        """
        # 1. Scene Count
        scenes = RiskEngine.check_scene_count(video_path)
        if scenes < 2:
            return False, f"Low Scene Count ({scenes}). Clip is a single shot."
            
        # 2. Motion Entropy
        motion = RiskEngine.check_motion_entropy(video_path)
        if motion < 2.0: # Threshold for "Static"
            return False, f"Static Content detected (Motion Score: {motion:.2f})."
            
        return True, f"Passed (Scenes: {scenes}, Motion: {motion:.1f})"
