"""
Decision Engine Module (Criterion 1 & 6)
----------------------------------------
Implements "Expected Value" (EV) logic to gate destructive actions.
EV = (P_success * Reward) - (P_failure * Cost)

Irreversibility Hierarchy (Cost):
- Metadata Update: 0.1
- Color Grading: 0.3
- Cropping: 0.6
- Inpainting (AI Fill): 0.9 (Highest Risk of Artifacts)
"""

import logging

logger = logging.getLogger("decision_engine")

class DecisionEngine:
    
    # Cost Constants (Risk)
    COST_METADATA = 0.1
    COST_COLOR = 0.3
    COST_CROP = 0.6
    COST_INPAINT = 0.9
    
    @staticmethod
    def calculate_ev(confidence: float, action_type: str, clip_value: float = 1.0) -> float:
        """
        Calculates Expected Value.
        Confidence: 0.0 to 1.0 (Probability of Success)
        Action Type: 'inpaint', 'crop', 'enhance'
        Clip Value: 0.0 to 1.0 (How valuable is this clip? Default 1.0)
        """
        
        # 1. Determine Cost based on Irreversibility
        cost = 0.5 # Default
        if action_type == "inpaint":
            cost = DecisionEngine.COST_INPAINT
        elif action_type == "crop":
            cost = DecisionEngine.COST_CROP
        elif action_type == "enhance":
            cost = DecisionEngine.COST_COLOR
        elif action_type == "metadata":
             cost = DecisionEngine.COST_METADATA
             
        # 2. Probability of Failure
        p_fail = 1.0 - confidence
        
        # 3. Reward (Value of Success)
        # If we succeed, we gain the 'clip_value'.
        reward = clip_value
        
        # 4. EV Formula
        # EV = (Prob_Success * Reward) - (Prob_Fail * Cost)
        ev = (confidence * reward) - (p_fail * cost)
        
        logger.debug(f"🧮 EV Calc: ({confidence:.2f} * {reward}) - ({p_fail:.2f} * {cost}) = {ev:.2f}")
        
        return ev

    @staticmethod
    def should_proceed(confidence: float, action_type: str, threshold: float = 0.0) -> bool:
        """
        Gatekeeper Function. Returns True if EV > threshold.
        """
        ev = DecisionEngine.calculate_ev(confidence, action_type)
        
        if ev > threshold:
            logger.info(f"✅ EV PASS ({ev:.2f} > {threshold}): Proceeding with {action_type}")
            return True
        else:
            logger.warning(f"⛔ EV FAIL ({ev:.2f} <= {threshold}): Skipping {action_type} (Risk too high)")
            return False

class StabilityAnalyst:
    """
    Implements the Weighted Stability Scoring Engine.
    Default Assumption: Watermark is STATIC.
    """
    
    # Weights
    W_IOU = 0.45
    W_POS = 0.25
    W_SCALE = 0.15
    W_AR = 0.10
    W_TIME = 0.05
    
    @staticmethod
    def calculate_score(trajectory: list, total_frames: int, im_w: int = 1920, im_h: int = 1080) -> float:
        """
        Computes STATIC_SCORE based on trajectory data.
        Returns: 0.0 to 1.0 (Higher = More Static).
        """
        if not trajectory: return 0.0
        
        import numpy as np
        
        # Extract metrics
        xs = [t['x'] for t in trajectory]
        ys = [t['y'] for t in trajectory]
        ws = [t['w'] for t in trajectory]
        hs = [t['h'] for t in trajectory]
        
        # 1. Position Stability
        # Rule: "Position variance < 5% frame size" (User Rule: 5%)
        # We use Range (Max-Min) as strictly bounded drift.
        range_x = max(xs) - min(xs)
        range_y = max(ys) - min(ys)
        
        limit_x = im_w * 0.05
        limit_y = im_h * 0.05
        
        pos_score = 0.0
        if range_x <= limit_x and range_y <= limit_y:
            pos_score = StabilityAnalyst.W_POS
                
        # 2. Scale Consistency
        # Rule: "Scale variance < 5%"
        w_var = (max(ws) - min(ws)) / np.mean(ws) if np.mean(ws) > 0 else 0
        h_var = (max(hs) - min(hs)) / np.mean(hs) if np.mean(hs) > 0 else 0
        
        scale_score = 0.0
        if w_var <= 0.05 and h_var <= 0.05:
            scale_score = StabilityAnalyst.W_SCALE
            
        # 3. Aspect Ratio Consistency
        # "Ratio variance <= 0.05 -> +0.10"
        ars = [w/h for w,h in zip(ws, hs)]
        ar_std = np.std(ars)
        ar_score = 0.0
        if ar_std <= 0.05:
            ar_score = StabilityAnalyst.W_AR
            
        # 4. Temporal Presence
        # "Visible in >= 80% of frames -> +0.05" (User Rule: 80%)
        presence = len(trajectory) / max(1, total_frames)
        time_score = 0.0
        if presence >= 0.80:
            time_score = StabilityAnalyst.W_TIME
            
        # 5. IoU Stability
        # "If IoU >= 0.70 across >= 60% frames -> +0.45"
        # Reference: Median Box
        med_x = np.median(xs)
        med_y = np.median(ys)
        med_w = np.median(ws)
        med_h = np.median(hs)
        
        stable_frames = 0
        for i in range(len(trajectory)):
            # IoU with Median
            tx, ty, tw, th = xs[i], ys[i], ws[i], hs[i]
            
            x1 = max(tx, med_x)
            y1 = max(ty, med_y)
            x2 = min(tx+tw, med_x+med_w)
            y2 = min(ty+th, med_y+med_h)
            
            inter_w = max(0, x2 - x1)
            inter_h = max(0, y2 - y1)
            intersection = inter_w * inter_h
            
            union = (tw*th) + (med_w*med_h) - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou >= 0.70:
                stable_frames += 1
                
        iou_score = 0.0
        if (stable_frames / len(trajectory)) >= 0.60:
            iou_score = StabilityAnalyst.W_IOU
            
        total_score = iou_score + pos_score + scale_score + ar_score + time_score
        
        logger.info(f"📊 Stability Score: {total_score:.2f} (IoU:{iou_score:.2f} Pos:{pos_score:.2f} Scale:{scale_score:.2f} AR:{ar_score:.2f} Time:{time_score:.2f})")
        return total_score
