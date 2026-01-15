"""
Text Region Detector (Helper for Composite Watermarks)
------------------------------------------------------
Scans a local neighborhood around a confirmed logo to find associated text.
Used to merge "Logo + Text" into a single Composite Watermark.

Rules:
- Only scans local ROI (Economy)
- Uses MSER + Morphological Closing (Accuracy)
- Strict aspect ratio and density filters (Safety)
- Conservative NMS (Suppression)

STRICT AUDIT COMPLIANT: Normalized MSER, Kernel Clamping, NMS, Stroke Density.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger("text_detector")

class TextRegionDetector:
    
    @staticmethod
    def find_nearby_text_candidates(frame: np.ndarray, logo_box: dict) -> list:
        """
        Scans neighborhood of logo_box for text-like regions.
        Returns list of boxes [{'x':.., 'y':.., 'w':.., 'h':..}]
        """
        candidates = []
        try:
            h_img, w_img = frame.shape[:2]
            lx, ly, lw, lh = int(logo_box['x']), int(logo_box['y']), int(logo_box['w']), int(logo_box['h'])
            
            # 1. Define Search Region (Local Neighborhood)
            # x ± 2.0 * w, y ± 1.5 * h
            # Safe ROI expansion
            pad_x = int(lw * 2.0)
            pad_y = int(lh * 1.5)
            
            search_x = max(0, lx - pad_x)
            search_y = max(0, ly - pad_y)
            search_w = min(w_img - search_x, lw + 2*pad_x)
            search_h = min(h_img - search_y, lh + 2*pad_y)
            
            # Crop ROI
            roi = frame[search_y:search_y+search_h, search_x:search_x+search_w]
            if roi.size == 0: return []
            roi_area = roi.shape[0] * roi.shape[1]
            if roi_area < 1000: return [] # Too small to process
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # 2. Text Detection Strategy: MSER (Robust for text)
            # Normalized Parameters
            # min_area: 0.1% of ROI or 5px
            # max_area: 30% of ROI (Text is rarely huge solid blocks)
            min_area = max(5, int(roi_area * 0.001))
            max_area = int(roi_area * 0.3)
            
            mser = cv2.MSER_create(max_area=max_area, min_area=min_area)
            regions, _ = mser.detectRegions(gray)
            
            # 3. Filter & Group Regions (Char Level)
            boxes = []
            for p in regions:
                x, y, w, h = cv2.boundingRect(p)
                
                # Aspect Ratio Filter (Strict Char Filter)
                # 0.3 (tall 'l') to 3.0 (wide 'm' or small word)
                aspect = w / h
                if not (0.3 <= aspect <= 3.0): continue
                
                # Height FilterRelative to Logo
                # Text should be roughly same scale or smaller than logo height
                # Allow up to 120% logo height (rare cases)
                if h > lh * 1.2: continue 
                
                boxes.append([x, y, x+w, y+h])
                
            if not boxes: return []
            
            boxes = np.array(boxes)
            
            # 4. Morphological Grouping (Merge chars into words)
            mask = np.zeros_like(gray)
            for box in boxes:
                cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), (255), -1)
                
            # Clamped Kernel Width ensures we don't bleed infinitely
            # Min 3px, Max 25px
            kernel_w_ideal = int(lw * 0.15)
            kernel_w = max(3, min(25, kernel_w_ideal))
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
            dilated = cv2.dilate(mask, kernel, iterations=2)
            
            # Find Words
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_candidates = []
            
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                
                # Analyze Word Candidate
                aspect = cw / ch
                
                # A word/link is usually wide
                if aspect < 1.0: continue # Single letters or noise
                
                # Density Check (Stroke Consistency)
                # Text is sparse. Solid blocks (density ~1.0) are usually graphics/bars.
                roi_word = mask[cy:cy+ch, cx:cx+cw]
                # Non-zero pixels in the *original char mask* vs bounding box area
                density = cv2.countNonZero(roi_word) / (cw * ch)
                
                # Valid text density: usually 0.25 to 0.85
                if not (0.20 <= density <= 0.90): continue
                
                # Translate to Global
                gx = search_x + cx
                gy = search_y + cy
                
                # 5. Overlap Logic (Logo Self-Recapture Bug)
                # Calculate Intersection with Logo Box
                intersect_x = max(lx, gx)
                intersect_y = max(ly, gy)
                intersect_w = min(lx+lw, gx+cw) - intersect_x
                intersect_h = min(ly+lh, gy+ch) - intersect_y
                
                if intersect_w > 0 and intersect_h > 0:
                    intersection_area = intersect_w * intersect_h
                    candidate_area = cw * ch
                    # If > 15% of candidate overlaps with logo, it's likely PART of the logo
                    if intersection_area / candidate_area > 0.15:
                        continue 
                
                raw_candidates.append([gx, gy, gx+cw, gy+ch])
                
            # 6. Non-Maximum Suppression (NMS)
            # Remove nested or highly overlapping boxes
            if not raw_candidates: return []
            
            raw_boxes = np.array(raw_candidates)
            # Score by area (prefer larger fused words)
            scores = (raw_boxes[:, 2] - raw_boxes[:, 0]) * (raw_boxes[:, 3] - raw_boxes[:, 1])
            
            # Standard NMS
            # Pick by score, suppress if IoU > threshold
            nms_indices = cv2.dnn.NMSBoxes(
                bboxes=[(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])) for b in raw_boxes],
                scores=scores.tolist(),
                score_threshold=0.0,
                nms_threshold=0.3
            )
            
            final_candidates = []
            if len(nms_indices) > 0:
                for i in nms_indices.flatten():
                     b = raw_boxes[i]
                     # Clamp to frame bounds (Out-of-bounds safety)
                     fx = max(0, int(b[0]))
                     fy = max(0, int(b[1]))
                     fw = min(w_img - fx, int(b[2] - b[0]))
                     fh = min(h_img - fy, int(b[3] - b[1]))
                     
                     if fw > 4 and fh > 4:
                         final_candidates.append({'x': fx, 'y': fy, 'w': fw, 'h': fh})
            
            if final_candidates:
                logger.info(f"   🧩 Found {len(final_candidates)} text regions (NMS filtered).")
                
            return final_candidates

        except Exception as e:
            logger.warning(f"Text detection failed: {e}")
            return []
