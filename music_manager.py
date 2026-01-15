
import os
import glob
import random
import logging
import json
from typing import List, Tuple, Dict
import subprocess

logger = logging.getLogger("music_manager")

class ContinuousMusicManager:
    """
    Manages a continuous music timeline for a batch of videos.
    State is specific to an instance (one compilation job).
    """
    def __init__(self, music_dir: str = "music"):
        self.music_dir = music_dir
        self.playlist = self._load_playlist()
        
        # State
        self.current_track_index = 0
        # Per-Track Cursor State (The "Bookmark" for each song)
        # { "music/song1.mp3": 15.0, "music/song2.mp3": 0.0 }
        self.track_offsets = {p: 0.0 for p in self.playlist}
        
        # Shuffle on init to ensure variety per compilation
        if self.playlist:
            random.shuffle(self.playlist)
            # Re-init offsets after shuffle just to be safe (keys match)
            self.track_offsets = {p: 0.0 for p in self.playlist}
        
        self.track_durations = {} # Cache

    def _load_playlist(self) -> List[str]:
        if not os.path.exists(self.music_dir):
            return []
        files = glob.glob(os.path.join(self.music_dir, "*.mp3")) + \
                glob.glob(os.path.join(self.music_dir, "*.wav"))
        return sorted(files) 

    def _get_duration(self, path: str) -> float:
        """Get duration with caching"""
        if path in self.track_durations:
            return self.track_durations[path]
            
        try:
             cmd = [
                 "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                 "-of", "default=noprint_wrappers=1:nokey=1", path
             ]
             res = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
             dur = float(res.decode().strip())
             self.track_durations[path] = dur
             return dur
        except Exception as e:
            logger.warning(f"Failed to get duration for {os.path.basename(path)}: {e}")
            return 30.0 # Safety default

    def allocate_music(self, needed_duration: float) -> List[Dict]:
        """
        Allocates music in a ROUND-ROBIN fashion.
        Clip 1 -> Track A (0-15s)
        Clip 2 -> Track B (0-15s)
        Clip 3 -> Track A (15-30s) <- Continues where it left off!
        """
        if not self.playlist:
            return []

        segments = []
        
        # 1. Select Track (Round Robin)
        track_path = self.playlist[self.current_track_index]
        
        # 2. Get Saved State
        current_offset = self.track_offsets.get(track_path, 0.0)
        total_track_dur = self._get_duration(track_path)
        
        # 3. Calculate Segments (Handle Loop if needed)
        remaining_on_track = total_track_dur - current_offset
        
        if remaining_on_track >= needed_duration:
            # Simple case: Fits in remainder
            segments.append({
                "path": track_path,
                "start": current_offset,
                "duration": needed_duration
            })
            # Update State
            self.track_offsets[track_path] += needed_duration
        else:
            # Wrap around case
            # Part 1: End of track
            segments.append({
                "path": track_path,
                "start": current_offset,
                "duration": remaining_on_track
            })
            
            # Part 2: Start from beginning
            needed_rest = needed_duration - remaining_on_track
            
            # If needed_rest is larger than full duration, we simplify and just loop max once or fail?
            # Ideally loop multiple times if needed, but for <60s clips, likely just one wrap.
            # We'll validly assume needed_rest < total_track_dur usually.
            
            segments.append({
                "path": track_path,
                "start": 0.0,
                "duration": needed_rest
            })
            
            # Update State
            self.track_offsets[track_path] = needed_rest

        # 4. Rotate Index for NEXT caller
        self.current_track_index = (self.current_track_index + 1) % len(self.playlist)
        
        return segments
