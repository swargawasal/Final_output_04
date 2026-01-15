"""
Music Intelligence Module
-------------------------
Heuristic-based music classifier to support smart audio mixing.
NO ML/Deep Learning. Uses ffmpeg stats and keyword analysis.
"""

import os
import subprocess
import logging
import re
import math

logger = logging.getLogger("music_intelligence")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")

def classify_music(file_path: str) -> tuple[str, float]:
    """
    Analyzes audio file and returns (Genre, Confidence).
    Matches compiler.py expectation: genre, confidence = classify_music(path)
    """
    if not os.path.exists(file_path):
        return "neutral", 0.0

    filename = os.path.basename(file_path).lower()
    
    # 1. Keyword Classification
    keywords = {
        "lofi": ["lofi", "chill", "relax", "study", "sleep", "dream", "ambient"],
        "mass": ["gym", "workout", "phonk", "sigma", "trap", "bass", "energetic", "motivation", "hard"],
        "classical": ["classical", "piano", "violin", "orchestral", "symphony"],
        "romantic": ["love", "romantic", "emotional", "sad", "breakup"],
        "pop": ["pop", "upbeat", "summer", "vlog", "happy", "party"],
        "high_energy": ["rock", "metal", "fast", "run"]
    }
    
    detected_genre = "neutral"
    confidence = 0.5
    
    for g, keys in keywords.items():
        if any(k in filename for k in keys):
            detected_genre = g
            confidence = 0.9
            break
            
    # 2. RMS / Volume Analysis fallback
    if detected_genre == "neutral":
        try:
            cmd = [
                FFMPEG_BIN, "-i", file_path, 
                "-af", "volumedetect", 
                "-f", "null", "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stderr
            match = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", output)
            if match:
                mean_vol = float(match.group(1))
                if mean_vol > -14.0:
                    detected_genre = "mass"
                    confidence = 0.6
                elif mean_vol < -24.0:
                    detected_genre = "lofi"
                    confidence = 0.6
        except:
            pass
            
    return detected_genre, confidence

def get_filter_graph(genre: str, target_duration: float) -> str:
    """
    Returns the FFMPEG Audio Filter Graph for the music stream.
    Matches compiler.py call: preset = get_filter_graph(genre, music_len)
    
    The filter should:
    1. Trim/Pad music
    2. Apply Fade In/Out
    3. Normalize Volume
    
    Note: 'adelay' is applied externally in compiler.py. 
    This logic handles the internal shaping of the music clip.
    """
    
    # Defaults
    fade_in = 0.5
    fade_out = 1.0
    vol = 0.4
    
    if genre in ["lofi", "romantic", "ambient", "classical"]:
        fade_in = 2.0
        vol = 0.35
    elif genre in ["mass", "pop", "hiphop", "high_energy"]:
        fade_in = 0.2
        vol = 0.55
        
    # Safety: Ensure fade doesn't exceed duration
    if target_duration < (fade_in + fade_out):
        fade_in = target_duration * 0.2
        fade_out = target_duration * 0.2
        
    # Compiler.py expects: `atrim=0:LEN,{preset}[mus]`
    # So we return the chain AFTER trim.
    
    filter_chain = (
        f"volume={vol},"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={target_duration-fade_out:.2f}:d={fade_out:.2f}"
    )
    
    return filter_chain
