"""
Policy Memory Module (Criterion 5)
----------------------------------
Stores strategies strategies (Policies) and tracks their real-world performance.
Automatically disables policies that fall below success thresholds.

STRICT AUDIT COMPLIANT: Atomic Writes, Hysteresis, State Safety.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any
import shutil
import tempfile
import time

logger = logging.getLogger("policy_memory")

POLICY_FILE = "policy_memory.json"

class PolicyMemory:
    _instance = None
    # _data removed from class level to prevent shared state
    
    # Defaults
    MIN_ATTEMPTS = 5 # Warmup period
    DISABLE_THRESHOLD = 0.40 # If success rate < 40%, disable
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PolicyMemory, cls).__new__(cls)
            cls._instance._data = {} # Instance-owned state
            cls._instance.load()
        return cls._instance
    
    def load(self):
        """
        Loads policy data with failure resilience.
        """
        if os.path.exists(POLICY_FILE):
            try:
                with open(POLICY_FILE, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self._data = json.loads(content)
                    else:
                        logger.warning("Policy memory file empty. Starting fresh.")
                        self._data = {}
            except Exception as e:
                # Do NOT wipe memory on read error if we have state, 
                # but valid init requires clean slate if file is bad.
                # Here we default to empty only if load fails explicitly.
                logger.error(f"Failed to load policy memory: {e}")
                # We do NOT overwrite self._data here if it already had something (rare in __init__)
                # But for safety in reload scenarios:
                if not self._data: self._data = {}
        else:
            self._data = {}
            
    def save(self):
        """
        Saves policy data ATOMICALLY.
        """
        try:
            # Write to temp file first
            fd, tmp_path = tempfile.mkstemp(dir=".", text=True)
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(self._data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno()) # Ensure write to disk
                
                # Atomic rename
                # On Windows, os.replace is atomic for files
                os.replace(tmp_path, POLICY_FILE)
            except Exception as e:
                os.close(fd) # Safety close if fdopen failed before context
                if os.path.exists(tmp_path): os.remove(tmp_path)
                raise e
                
        except Exception as e:
            logger.error(f"Failed to save policy memory: {e}")

    def _sanitize_name(self, name: str) -> str:
        """Normalize policy name."""
        if not name: return "unknown_policy"
        clean = str(name).strip()
        return clean[:64]

    def get_policy(self, name: str) -> Dict[str, Any]:
        safe_name = self._sanitize_name(name)
        
        if safe_name not in self._data:
            # Initialize new policy
            self._data[safe_name] = {
                "name": safe_name,
                "attempts": 0,
                "success": 0,
                "fail": 0,
                "success_rate": 1.0, # Optimistic start
                "disabled": False,
                "last_updated": datetime.now().isoformat()
            }
        return self._data[safe_name]

    def update_policy(self, name: str, success: bool):
        """
        Updates policy stats and re-evaluates enabled status.
        """
        safe_name = self._sanitize_name(name)
        p = self.get_policy(safe_name)
        
        p["attempts"] += 1
        if success:
            p["success"] += 1
        else:
            p["fail"] += 1
            
        # Recalc Rate (Guard Division)
        if p["attempts"] > 0:
            p["success_rate"] = p["success"] / p["attempts"]
        else:
            p["success_rate"] = 0.0 # Should not happen
            
        p["success_rate"] = max(0.0, min(1.0, p["success_rate"]))
        p["last_updated"] = datetime.now().isoformat()
        
        # Check Disablement Rule
        # Hysteresis:
        # Disable if < THRESHOLD
        # Enable if > THRESHOLD + 0.2 AND attempts > Double Minimal (proven record)
        
        if p["attempts"] >= self.MIN_ATTEMPTS:
            
            # Logic A: Should we Disable?
            if not p["disabled"]:
                if p["success_rate"] < self.DISABLE_THRESHOLD:
                    logger.warning(f"🚫 POLICY DISABLED: '{safe_name}' (Rate: {p['success_rate']:.2f} < {self.DISABLE_THRESHOLD})")
                    p["disabled"] = True
            
            # Logic B: Should we Re-enable?
            else:
                # Require stronger proof to come back
                # This prevents flip-flopping near the threshold
                recovery_threshold = self.DISABLE_THRESHOLD + 0.2
                significant_attempts = self.MIN_ATTEMPTS * 2
                
                if p["success_rate"] >= recovery_threshold and p["attempts"] >= significant_attempts:
                     logger.info(f"✅ POLICY RE-ENABLED: '{safe_name}' (Rate: {p['success_rate']:.2f})")
                     p["disabled"] = False
                     
        self.save()
        
    def is_enabled(self, name: str) -> bool:
        """
        Returns True if policy is allowed to run.
        """
        safe_name = self._sanitize_name(name)
        p = self.get_policy(safe_name)
        
        # Optimistic Start Check:
        # If we are in warmup period (attempts < MIN), we should generally be enabled 
        # unless manual intervention disabled it (which we respect).
        # But 'disabled' defaults to False, so we are good.
        
        if p["disabled"]:
            # Don't log spam, but maybe log once per run?
            # For now, keep it silent or debug
            return False
            
        return True

    def get_success_rate(self, name: str) -> float:
        """Helper to get rate safely."""
        p = self.get_policy(name)
        return p.get("success_rate", 0.0)

# Global Instance
policy_db = PolicyMemory()
