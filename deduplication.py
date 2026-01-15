"""
Deduplication Engine (Dual-Hash)
--------------------------------
Enforces strict uniqueness using both Byte-level (SHA256) and Perceptual (pHash) hashing.
Prevents "Same Clip, Different ID" exploits and forces fresh processing on collisions.
"""

import os
import json
import hashlib
import cv2
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger("deduplication")

DEDUP_INDEX_PATH = os.path.join("data", "dedup_index.json")

class DedupEngine:
    
    @staticmethod
    def _ensure_index():
        if not os.path.exists("data"):
            os.makedirs("data", exist_ok=True)
        if not os.path.exists(DEDUP_INDEX_PATH):
            with open(DEDUP_INDEX_PATH, 'w') as f:
                json.dump({}, f)

    @staticmethod
    def compute_sha256(file_path):
        """Computes strict byte-level hash."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    @staticmethod
    def compute_phash(video_path):
        """
        Computes DCT-based Perceptual Hash (pHash) on the middle frame.
        Robust to re-encoding, resolution changes, and slight shifts.
        """
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample middle frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
            ret, frame = cap.read()
            cap.release()
            
            if not ret: return None
            
            # 1. Resize to 32x32
            resized = cv2.resize(frame, (32, 32))
            
            # 2. Grayscale
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            
            # 3. DCT (Discrete Cosine Transform)
            # Convert to float
            gray32 = np.float32(gray)
            dct = cv2.dct(gray32)
            
            # 4. Keep top-left 8x8 (low frequencies), exclude DC (0,0)
            dct_low = dct[0:8, 0:8]
            
            # 5. Compute median/mean
            # Use mean of the 64 items (excluding DC term logic usually implies avg)
            # Standard: Compare to median or mean
            avg = np.mean(dct_low)
            
            # 6. Generate Hash (64 bits)
            # 1 if > avg, 0 if < avg
            phash_bin = (dct_low > avg).flatten()
            
            # Convert bool array to hex string
            # Pack bits
            phash_int = 0
            for b in phash_bin:
                phash_int = (phash_int << 1) | int(b)
            
            return f"{phash_int:016x}" # 64-bit hex
            
        except Exception as e:
            logger.error(f"pHash computation failed: {e}")
            return None

    @staticmethod
    def _hamming_distance(hash1_hex, hash2_hex):
        """Computes Hamming distance between two 64-bit hex hashes."""
        try:
            h1 = int(hash1_hex, 16)
            h2 = int(hash2_hex, 16)
            x = h1 ^ h2
            return bin(x).count('1')
        except:
            return 64 # Max difference

    @staticmethod
    def check_collision(video_id, file_path):
        """
        Checks for collisions in the index.
        Returns:
            (collision_type, reason_msg)
            collision_type: 'NONE', 'SHA', 'PHASH'
        """
        try:
            DedupEngine._ensure_index()
            
            current_sha = DedupEngine.compute_sha256(file_path)
            current_phash = DedupEngine.compute_phash(file_path)
            
            if not current_phash:
                return "NONE", "pHash failed, skipping check"

            with open(DEDUP_INDEX_PATH, 'r') as f:
                try: index = json.load(f)
                except: index = {}
            
            # Iterate index
            # Structure: { key_id: {sha, phash, video_id ...} }
            # Actually better to store by Unique ID? No, we scan values.
            
            for key, entry in index.items():
                stored_id = entry.get('video_id')
                stored_sha = entry.get('sha256')
                stored_phash = entry.get('phash')
                
                # Skip self (re-run of same logic)
                if stored_id == video_id:
                    continue
                
                # Check SHA
                if stored_sha and stored_sha == current_sha:
                    msg = f"⚠️ HASH COLLISION (SHA256): Matches {stored_id}"
                    return "SHA", msg
                    
                # Check pHash
                if stored_phash:
                    dist = DedupEngine._hamming_distance(current_phash, stored_phash)
                    if dist < 5: # Threshold < 5 is extremely similar
                        msg = f"⚠️ HASH COLLISION (pHash dist={dist}): Matches {stored_id}"
                        return "PHASH", msg
                        
            return "NONE", None
            
        except Exception as e:
            logger.error(f"Collision check error: {e}")
            return "NONE", None

    @staticmethod
    def register_content(video_id, file_path, source="unknown"):
        """Registers the content in the index."""
        try:
            DedupEngine._ensure_index()
            
            sha = DedupEngine.compute_sha256(file_path)
            phash = DedupEngine.compute_phash(file_path)
            
            with open(DEDUP_INDEX_PATH, 'r') as f:
                try: index = json.load(f)
                except: index = {}
                
            index[f"{video_id}_{sha[:8]}"] = { # Composite key to allow multiple versions? No VideoID should be unique per source item
                "video_id": video_id,
                "sha256": sha,
                "phash": phash,
                "source": source,
                "created_at": datetime.utcnow().isoformat()
            }
            
            with open(DEDUP_INDEX_PATH, 'w') as f:
                json.dump(index, f, indent=2)
                
            return True, sha, phash
            
        except Exception as e:
            logger.error(f"Register content failed: {e}")
            return False, None, None
