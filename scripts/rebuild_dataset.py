import os
import sys
import glob
import cv2
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from opencv_watermark import WatermarkLearner, DATASET_FILE, POSITIVE_DIR, NEGATIVE_DIR, CSV_FIELDS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rebuild_dataset")

def rebuild():
    logger.info("♻️ Rebuilding Watermark Dataset from Templates...")
    
    # 1. Backup old dataset
    if os.path.exists(DATASET_FILE):
        os.rename(DATASET_FILE, DATASET_FILE + ".bak")
        logger.info(f"📦 Backed up old dataset to {DATASET_FILE}.bak")
        
    # 2. Process Positive Templates
    pos_files = glob.glob(os.path.join(POSITIVE_DIR, "*.png"))
    logger.info(f"➕ Processing {len(pos_files)} positive templates...")
    
    for f in pos_files:
        try:
            img = cv2.imread(f)
            if img is None: continue
            
            h, w = img.shape[:2]
            # Mock coordinates (we assume the template IS the watermark)
            # But extract_features expects coordinates relative to a larger frame.
            # For training data, we can treat the template AS the frame and the crop AS the whole thing.
            # OR, we should pass the template image as the 'frame' and a full-size rect as 'coords'.
            
            coords = {'x': 0, 'y': 0, 'w': w, 'h': h}
            
            # Mock match data (since we know it's a template)
            match_data = {
                "kp_count": 50, # Dummy high value
                "template_pos_score": 1.0,
                "template_neg_score": 0.0
            }
            
            features = WatermarkLearner.extract_features(img, coords, match_data)
            if features:
                WatermarkLearner.log_feedback(features, label=1)
                
        except Exception as e:
            logger.warning(f"Failed to process {f}: {e}")

    # 3. Process Negative Templates
    neg_files = glob.glob(os.path.join(NEGATIVE_DIR, "*.png"))
    logger.info(f"➖ Processing {len(neg_files)} negative templates...")
    
    for f in neg_files:
        try:
            img = cv2.imread(f)
            if img is None: continue
            
            h, w = img.shape[:2]
            coords = {'x': 0, 'y': 0, 'w': w, 'h': h}
            
            match_data = {
                "kp_count": 10, # Dummy low value
                "template_pos_score": 0.0,
                "template_neg_score": 1.0
            }
            
            features = WatermarkLearner.extract_features(img, coords, match_data)
            if features:
                WatermarkLearner.log_feedback(features, label=0)
                
        except Exception as e:
            logger.warning(f"Failed to process {f}: {e}")
            
    logger.info("✅ Dataset Rebuild Complete!")

if __name__ == "__main__":
    rebuild()
