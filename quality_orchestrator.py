"""
Visual Safety & Quality Orchestrator
------------------------------------
Governs spatial effects (Vignette) based on Face/Watermark geometry.
Strict adherence to User Ruleset #65.

Role:
- Face Detection (OpenCV DNN)
- Geometry Mapping (Shared Space)
- Policy Decision (Vignette Allowed?)

Output: Structured Decision Object (JSON-compatible dict)
"""

import cv2
import numpy as np
import os
import logging

logger = logging.getLogger("quality_orchestrator")

class HumanPresenceGuard:
    def __init__(self):
        self.face_net = None
        self._load_face_model()

    def _load_face_model(self):
        """Loads OpenCV DNN Face Detector (ResNet-10)"""
        try:
            proto = "models/deploy.prototxt"
            model = "models/res10_300x300_ssd_iter_140000.caffemodel"
            
            if os.path.exists(proto) and os.path.exists(model):
                 self.face_net = cv2.dnn.readNetFromCaffe(proto, model)
                 logger.info("✅ HumanGuard: Loaded DNN Identity Detector")
            else:
                 logger.warning("⚠️ HumanGuard: DNN Models not found. Assuming NO HUMANS (CAUTION).")
                 self.face_net = None
        except Exception as e:
            logger.error(f"HumanGuard Init Error: {e}")
            self.face_net = None

    def detect_faces(self, frame):
        """
        Returns list of faces: {'box': [x,y,w,h], 'confidence': float}
        STRICT: Only returns faces with confidence >= 0.6
        """
        if self.face_net is None: return []

        h_img, w_img = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence < 0.6: continue
                
            box = detections[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
            (x1, y1, x2, y2) = box.astype("int")
            
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img-1, x2); y2 = min(h_img-1, y2)
            
            w = x2 - x1
            h = y2 - y1
            
            if w > 0 and h > 0:
                faces.append({
                    'box': [x1, y1, w, h],
                    'confidence': float(confidence)
                })
        
        return faces

    def analyze_human_presence(self, frame_path: str) -> dict:
        """
        Primary Quality Signal:
        Detects if humans are present to GATE risky enhancements.

        Returns:
            {
              "has_humans": bool,
              "safety_level": "SAFE_SCENERY" | "CAUTION_HUMAN" | "UNKNOWN"
            }
        """
        try:
            if self.face_net is None:
                return {"has_humans": False, "safety_level": "UNKNOWN"}

            frame = cv2.imread(frame_path)
            if frame is None:
                return {"has_humans": False, "safety_level": "UNKNOWN"}
                
            faces = self.detect_faces(frame)
            
            if faces:
                # Human detected -> Enforce constraints
                return {
                    "has_humans": True,
                    "safety_level": "CAUTION_HUMAN"
                }
            else:
                # No human -> Allow stronger processing
                return {
                    "has_humans": False, 
                    "safety_level": "SAFE_SCENERY"
                }

        except Exception as e:
            logger.error(f"Human Guard Failed: {e}")
            # Fail-safe: Assume humans exist to be safe? Or unknown?
            # "OpenCV DNN exists to Protect humans" -> If error, assume Human to be safe.
            return {"has_humans": True, "safety_level": "CAUTION_FAILSAFE"}

# Singleton
human_guard = HumanPresenceGuard()
