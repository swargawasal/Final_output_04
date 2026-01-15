"""
Quality Evaluator Module
------------------------
Implements "Blind Quality Evaluation" (Criterion 4) to verify actions.
Compares before/after video quality to detect destructive failures.

Metrics:
1. Blur Delta (Laplacian Variance): Detects if the image became significantly blurrier.
2. Edge Density Delta: Detects loss of texture/detail (e.g., over-smoothing).

STRICT AUDIT COMPLIANT: Adaptive Thresholds, Frame Sync, Worst-Case Scoring.
"""

import cv2
import numpy as np
import logging
import os

logger = logging.getLogger("quality_evaluator")

class QualityEvaluator:
    """
    Evaluates the quality impact of a video transformation.
    """
    
    # Thresholds for "FAIL"
    # We use Variance/Edge delta as proxy for Quality Drop.
    # Scores are usually negative (metrics drop).
    BLUR_THRESHOLD = -0.15  # Soft fail if blur metric drops >15%
    EDGE_THRESHOLD = -0.15  # Soft fail if edge metric drops >15%
    HARD_FAIL_THRESHOLD = -0.45 # Hard fail if any metric drops >45%
    
    @staticmethod
    def _get_metrics(frame):
        if frame is None: return 0.0, 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur (Laplacian Variance) - Higher is sharper
        # Normalize by max pixel value squared (255^2) to make it scale-invariant? 
        # Actually, standard variance is fine if we compare same-scale images.
        # But we should handle brightness.
        # Let's keep variance but ensure we don't divide by zero later.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # 2. Edge Density (Adaptive Canny) - Higher is more textured
        # Adaptive thresholds based on median intensity
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(gray, lower, upper)
        edge_density = np.count_nonzero(edges) / edges.size
        
        return variance, edge_density

    @staticmethod
    def evaluate_quality(before_path: str, after_path: str) -> dict:
        """
        Compares input vs output video quality.
        Returns dict with deltas and PASS/FAIL status.
        """
        cap_a = None
        cap_b = None
        try:
            if not os.path.exists(before_path) or not os.path.exists(after_path):
                return {"status": "ERROR", "reason": "Missing files", "score": -1.0, "deltas": {}, "reasons": ["Missing files"]}
            
            cap_a = cv2.VideoCapture(before_path)
            cap_b = cv2.VideoCapture(after_path)
            
            if not cap_a.isOpened() or not cap_b.isOpened():
                 return {"status": "ERROR", "reason": "Cannot open video", "score": -1.0, "deltas": {}, "reasons": ["IO Error"]}
            
            # 1. Frame Sync & Count Check
            count_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
            count_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if count_a <= 0 or count_b <= 0:
                 return {"status": "ERROR", "reason": "Empty video", "score": -1.0, "deltas": {}, "reasons": ["Empty Video"]}
                 
            # Structural Mismatch Check
            count_diff = abs(count_a - count_b)
            if count_diff / max(count_a, 1) > 0.10: # >10% frame count diff
                 return {
                     "status": "HARD_FAIL", 
                     "reason": f"Frame Mismatch ({count_a} vs {count_b})",
                     "score": -1.0,
                     "deltas": {}, 
                     "reasons": ["Significant Frame Desync"]
                 }
            
            # 2. Dynamic Sampling
            # Sample between 3 and 7 frames
            min_dim = min(count_a, count_b)
            num_samples = max(3, min(7, int(min_dim / 30))) # e.g. 150 frames -> 5 samples
            indices = np.linspace(0, min_dim - 1, num_samples, dtype=int).tolist()
            
            # Remove duplicates if video is very short
            indices = sorted(list(set(indices)))
            
            var_deltas = []
            edge_deltas = []
            
            epsilon = 1e-6 # float noise floor
            
            for idx in indices:
                cap_a.set(cv2.CAP_PROP_POS_FRAMES, idx)
                cap_b.set(cv2.CAP_PROP_POS_FRAMES, idx)
                
                ret_a, frame_a = cap_a.read()
                ret_b, frame_b = cap_b.read()
                
                if not ret_a or not ret_b: continue
                
                # Resize B to match A if needed (Handle Resolution Changes)
                if frame_a.shape != frame_b.shape:
                    frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
                
                v_a, e_a = QualityEvaluator._get_metrics(frame_a)
                v_b, e_b = QualityEvaluator._get_metrics(frame_b)
                
                # Calculate % Change with noise floor
                # If baseline is near zero, delta is unstable -> skip or clamp
                if v_a < epsilon: v_delta = 0.0 
                else: v_delta = (v_b - v_a) / v_a
                    
                if e_a < epsilon: e_delta = 0.0
                else: e_delta = (e_b - e_a) / e_a
                
                var_deltas.append(v_delta)
                edge_deltas.append(e_delta)
                
            if not var_deltas:
                return {"status": "ERROR", "reason": "No valid frames analyzed", "score": -1.0, "deltas": {}, "reasons": ["No Frames"]}
            
            avg_var_delta = np.mean(var_deltas)
            avg_edge_delta = np.mean(edge_deltas)
            
            # 3. Worst-Case Scoring
            # Instead of average, we take the minimum performance
            score = min(avg_var_delta, avg_edge_delta)
            
            reasons = []
            status = "PASS"
            
            # 4. Strict Pass Logic
            if score < QualityEvaluator.HARD_FAIL_THRESHOLD:
                status = "HARD_FAIL"
                reasons.append(f"Catastrophic Degrade (Score: {score:.2f})")
            
            elif avg_var_delta < QualityEvaluator.BLUR_THRESHOLD:
                status = "SOFT_FAIL"
                reasons.append(f"Blurry ({avg_var_delta:.2f})")
                
            elif avg_edge_delta < QualityEvaluator.EDGE_THRESHOLD:
                status = "SOFT_FAIL"
                reasons.append(f"Detail Lost ({avg_edge_delta:.2f})")
            
            if status != "PASS":
                 logger.warning(f"⚠️ Quality {status}: Score={score:.3f} | Blur={avg_var_delta:.3f}, Edge={avg_edge_delta:.3f}")
            else:
                 logger.info(f"✅ Quality PASS: Score={score:.3f}")

            return {
                "status": status,
                "score": round(float(score), 3),
                "reasons": reasons,
                "deltas": {
                    "blur": round(float(avg_var_delta), 3), 
                    "edge": round(float(avg_edge_delta), 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Quality Eval Failed: {e}")
            return {"status": "ERROR", "reason": str(e), "score": -1.0, "deltas": {}, "reasons": ["exception"]}
            
        finally:
            if cap_a: cap_a.release()
            if cap_b: cap_b.release()
