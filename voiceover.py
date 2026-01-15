"""
AI Voiceover Generator
----------------------
Uses gTTS (Google Text-to-Speech) to generate micro-commentary.
Strictly limited scope: Short, optional, non-blocking additions.

STRICT AUDIT COMPLIANT: Atomic Writes, Threaded Timeout, Smart Filters.
"""

import os
import logging
import random
import threading
import hashlib
import re
import shutil
import time
import tempfile
from typing import Optional, Dict, Any

logger = logging.getLogger("voiceover")

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False
    logger.warning("⚠️ gTTS not installed. Voiceover will be disabled.")

try:
    import edge_tts
    import asyncio
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False
    logger.warning("⚠️ edge-tts not installed. Voiceover Quality degraded.")

class VoiceoverGenerator:
    def __init__(self):
        self.enabled = os.getenv("ENABLE_MICRO_VOICEOVER", "yes").lower() == "yes"
        self.lang = "en"
        
        # 2. Configurable Env Vars
        self.min_chars = int(os.getenv("VOICEOVER_MIN_CHARS", 5))
        self.max_chars = int(os.getenv("VOICEOVER_MAX_CHARS", 200))
        self.tld_overrides = os.getenv("VOICEOVER_TLDS", "").split(",") if os.getenv("VOICEOVER_TLDS") else []
        self.slow_mode = os.getenv("VOICEOVER_SLOW_MODE", "no").lower() == "yes"
        self.safe_ascii = os.getenv("VOICEOVER_SAFE_ASCII_ONLY", "no").lower() == "yes"
        self.timeout = int(os.getenv("VOICEOVER_TIMEOUT", 15))
        self.smart_filter = os.getenv("VOICEOVER_SMART_FILTER", "no").lower() == "yes"
        
        # Accents (com, co.uk, us, ca, co.in, etc.)
        self.accents = [x.strip() for x in self.tld_overrides if x.strip()]
        if not self.accents:
            self.accents = ["com", "co.uk", "us", "ca", "co.in", "com.au"]
            
        # State
        self._last_meta: Dict[str, Any] = {}
        self._debug: Dict[str, Any] = {}

    def _sanitize_text(self, text: str) -> str:
        """3. Sanitize Input Text"""
        if not text: return ""
        
        # Collapse spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Control chars
        text = "".join(ch for ch in text if ch.isprintable())
        
        if self.safe_ascii:
            # Remove non-ascii
            text = text.encode('ascii', 'ignore').decode('ascii')
            
        # Truncate to max_chars (preserve words if possible)
        if len(text) > self.max_chars:
            cut = text[:self.max_chars]
            last_space = cut.rfind(" ")
            if last_space > self.max_chars // 2:
                 text = cut[:last_space] + "..."
            else:
                 text = cut + "..."
                 
        return text

    def _is_nonsense(self, text: str) -> bool:
        """9. Filter nonsense text (repeated chars, no vowels)."""
        if not text: return True
        # Check vowel presence
        if not re.search(r'[aeiouAEIOU]', text):
            return True
        # Check repetition (e.g. "aaaaa")
        if re.search(r'(.)\1{4,}', text):
            return True
        return False
        
    def _is_filler(self, text: str) -> bool:
        """Smart Filter: Reject basic filler text."""
        if not self.smart_filter: return False
        
        # Reject 1 word items that aren't powerful
        words = text.split()
        if len(words) <= 1:
            return True
            
        # Reject generic openings
        lower = text.lower()
        if lower.startswith("caption:") or lower.startswith("audio:"):
            return True
            
        return False

    def _get_deterministic_tld(self, text: str) -> str:
        """4. Deterministic Accent Selection"""
        seed_env = os.getenv("VOICEOVER_SEED")
        if seed_env:
            # Global deterministic seed
            # Use local RNG to preserve thread safety and not mess global state
            rng = random.Random(seed_env)
            return rng.choice(self.accents)
        
        # Per-text deterministic hash
        hash_val = int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16)
        idx = hash_val % len(self.accents)
        return self.accents[idx]
        
    def _generate_worker(self, text, tld, temp_path, result_container):
        """Worker thread logic."""
        try:
            # 1. Try EdgeTTS (Neural) First
            # Priority: High Quality
            if HAS_EDGE_TTS:
                try:
                    voice = "en-US-AriaNeural" # Default Female
                    # Maybe map TLD to voices?
                    if tld == "co.uk": voice = "en-GB-SoniaNeural"
                    elif tld == "com.au": voice = "en-AU-NatashaNeural"
                    elif tld == "co.in": voice = "en-IN-NeerjaNeural"
                    
                    async def _run_edge():
                        communicate = edge_tts.Communicate(text, voice)
                        await communicate.save(temp_path)
                        
                    asyncio.run(_run_edge())
                    
                    if os.path.exists(temp_path) and os.path.getsize(temp_path) > 100:
                        result_container['success'] = True
                        result_container['used_tld'] = f"edge-{voice}"
                        return
                except Exception as e_edge:
                    logger.warning(f"⚠️ EdgeTTS Failed ({e_edge}). Fallback to gTTS...")

            # 2. Fallback to gTTS (Robotic)
            if not HAS_GTTS:
                raise ImportError("No TTS engine available (gTTS missing, EdgeTTS failed/missing)")

            # Try chosen TLD
            try:
                tts = gTTS(text=text, lang=self.lang, tld=tld, slow=self.slow_mode)
                tts.save(temp_path)
                result_container['success'] = True
                result_container['used_tld'] = tld
                return
            except Exception as e:
                # Fallback to 'com' if different
                if tld != 'com':
                    logger.warning(f"⚠️ gTTS TLD '{tld}' failed, trying 'com': {e}")
                    tts = gTTS(text=text, lang=self.lang, tld='com', slow=self.slow_mode)
                    tts.save(temp_path)
                    result_container['success'] = True
                    result_container['used_tld'] = 'com'
                else:
                    raise e
                    
        except Exception as e:
            result_container['error'] = str(e)
            result_container['success'] = False

    def generate_voiceover(self, text: str, output_path: str) -> bool:
        """
        Fail-Safe Voiceover Generation with Timeout.
        Returns True if successful, False if ANY failure occurs.
        """
        temp_path = None
        try:
            if not self.enabled: return False
            if not HAS_GTTS: return False
            
            safe_text = self._sanitize_text(text)
            if len(safe_text) < self.min_chars: return False
            
            # Apply Filters
            if self._is_nonsense(safe_text):
                logger.info("ℹ️ Voiceover skipped: Nonsense detected.")
                return False
            if self._is_filler(safe_text):
                logger.info("ℹ️ Voiceover skipped: Filler detected.")
                return False
            
            # Ensure directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Prepare Atomic Temp
            fd, temp_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            
            tld = self._get_deterministic_tld(safe_text)
            
            # Generate in Thread with Timeout
            result = {'success': False, 'error': None}
            t = threading.Thread(target=self._generate_worker, args=(safe_text, tld, temp_path, result))
            t.daemon = True
            t.start()
            
            t.join(timeout=self.timeout)
            
            if t.is_alive():
                logger.error(f"❌ Voiceover timed out (> {self.timeout}s).")
                return False
            
            if not result['success']:
                logger.warning(f"⚠️ Voiceover generation failed: {result.get('error')}")
                return False
                
            # Validate Output
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) < 1024:
                logger.warning("⚠️ Voiceover file invalid (too small).")
                return False
                
            # Atomic Move
            if os.path.exists(output_path):
                try: os.remove(output_path)
                except: pass
            
            shutil.move(temp_path, output_path)
            
            # Meta
            self._last_meta = {
                "text_len": len(safe_text),
                "tld": result.get('used_tld'),
                "success": True
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Voiceover System Error: {e}")
            return False
        finally:
            # Cleanup temp
            if temp_path and os.path.exists(temp_path):
                try: os.remove(temp_path)
                except: pass

# Global Instance
voice_engine = VoiceoverGenerator()

def generate_voiceover(text: str, output_path: str) -> bool:
    return voice_engine.generate_voiceover(text, output_path)
