import os
import logging
import subprocess

import json
import time
import threading

logger = logging.getLogger("audio_processing")

# Configuration & Safety Constants
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
ENABLE_TREMOLO = os.getenv("ENABLE_TREMOLO", "no").lower() == "yes"
COMPILATION_MASTER_MODE = os.getenv("COMPILATION_MASTER_MODE", "heavy")
AUDIO_TIMEOUT = int(os.getenv("AUDIO_TIMEOUT", 25))
AUDIO_FORCE_SAFE = os.getenv("AUDIO_FORCE_SAFE", "no").lower() == "yes"
AUDIO_FORCE_HEAVY = os.getenv("AUDIO_FORCE_HEAVY", "no").lower() == "yes"
SPEECH_MODE = os.getenv("SPEECH_MODE", "auto").lower()

# Internal State (Soft Metadata)
_audio_meta = {
    "detected_reuse": False, # Assumed True for some inputs, can be set externally
    "transformative_applied": False,
    "effects_used": [],
    "speech_detected": False,
    "final_lufs_target": -14
}
_audio_debug = {"errors": []}

def _get_loudnorm_filter():
    """Returns the standard YouTube-safe loudness normalization filter."""
    return "loudnorm=I=-14:TP=-1.5:LRA=11"

def _safe_ffmpeg_run(cmd, timeout=AUDIO_TIMEOUT):
    """Robust wrapper for FFmpeg calls with strict timeout."""
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Audio operation timed out ({timeout}s)!")
        _audio_debug["errors"].append("timeout")
        return False
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode() if e.stderr else str(e)
        logger.warning(f"⚠️ FFmpeg Error: {err[:200]}...")
        _audio_debug["errors"].append(err)
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected Audio Crash: {e}")
        _audio_debug["errors"].append(str(e))
        return False

def heavy_remix(input_path: str, output_path: str, original_volume: float = 1.15) -> bool:
    """
    AUDIO REMIX FIX — Clean Beat Preset
    Removes noise, adds bass/treble boost, compression, and volume.
    
    Args:
        input_path: Input audio/video file
        output_path: Output audio file
        original_volume: Volume multiplier for original audio (default 1.15)
    """
    logger.info(f"🎛️ Audio Remix Fix: {input_path} (Volume: {original_volume}x)")
    
    # 1. Transformative Decision Layer
    # Logic: If reuse detected (default assumption for this module), ensure transform
    transform_required = True 
    _audio_meta["detected_reuse"] = True
    
    # 4. Smart Speech Protection
    is_speech_heavy = False
    if SPEECH_MODE == "yes":
         is_speech_heavy = True
    elif SPEECH_MODE == "auto":
         # Heuristic: Speech often has silence gaps > 0.5s or specific structure
         # We use our helper.
         if detect_silence(input_path):
              is_speech_heavy = True
              _audio_meta["speech_detected"] = True
              logger.info("🗣️ Speech detected (Auto-Protection Enabled)")
    
    if AUDIO_FORCE_SAFE: is_speech_heavy = True # Override
    
    # Effects Chain Construction
    effects = []
    
    # Pitch Shift (Transformative) - SKIP if speech heavy to avoid chipmunk
    if not is_speech_heavy and not AUDIO_FORCE_SAFE:
         effects.append("atempo=1.03") # Tempo shift is safer than pitch for speech
         _audio_meta["effects_used"].append("tempo_shift")
    
    # EQ (Bass/Treble)
    effects.append("equalizer=f=80:t=h:w=100:g=3")
    effects.append("equalizer=f=12000:t=h:w=2000:g=2")
    
    # Compression (Glue)
    effects.append("acompressor=threshold=-14dB:ratio=2.5:attack=20:release=200")
    
    # Volume Boost
    effects.append(f"volume={original_volume}")
    
    # Optional Tremolo (Non-Linear)
    if ENABLE_TREMOLO and not is_speech_heavy:
        effects.append("tremolo=f=1:d=0.4")
        _audio_meta["effects_used"].append("tremolo")
    
    # Limiter
    effects.append("alimiter=limit=0.95")
    
    # 3. Loudness Normalization (Mandatory)
    effects.append(_get_loudnorm_filter())
    
    af_filter = ",".join(effects)
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-af", af_filter,
        "-vn", 
        "-ac", "2", "-ar", "44100",
        output_path
    ]
    
    # 2. Hardened FFmpeg Safety Layer
    success = _safe_ffmpeg_run(cmd)
    
    if success:
        logger.info("✅ Audio Remix Complete (Clean Beat Preset + Loudnorm)")
        _audio_meta["transformative_applied"] = True
        return True
    
    # ERROR HANDLING / FALLBACKS
    logger.warning("⚠️ Complex remix failed. Attempting fallback chains...")
    
    # Fallback 1: LITE Chain (No tremolo, simple eq, loudness)
    fallback_effects = [
        f"volume={original_volume}",
        "alimiter=limit=0.95",
        _get_loudnorm_filter()
    ]
    cmd_lite = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-af", ",".join(fallback_effects),
        "-vn", "-ac", "2", "-ar", "44100",
        output_path
    ]
    
    if _safe_ffmpeg_run(cmd_lite):
        logger.info("✅ Audio Remix Complete (Lite Fallback)")
        return True
        
    # Fallback 2: Pass-Through (Safety Net)
    logger.error("❌ All remix attempts failed. Copying audio safely (Pass-Through).")
    cmd_copy = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-vn", "-acodec", "copy",
        output_path
    ]
    # We allow this to run directly as last resort
    try:
        subprocess.run(cmd_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except:
        return False

def mix_background_music(input_video: str, output_video: str, volume: float = 0.15) -> bool:
    """
    Mixes a random background track from 'music/' folder into the video.
    """
    import random
    import glob
    
    music_dir = "music"
    if not os.path.exists(music_dir):
        logger.warning(f"⚠️ Music directory '{music_dir}' not found.")
        return False
        
    tracks = glob.glob(os.path.join(music_dir, "*.mp3"))
    if not tracks:
        logger.warning(f"⚠️ No mp3 tracks found in '{music_dir}'.")
        return False
        
    bg_track = random.choice(tracks)
    logger.info(f"🎵 Mixing background music: {os.path.basename(bg_track)} (Vol: {volume})")
    
    # 5. Smart Mixer Hardening
    # - Stream loop the music input
    # - Dynamic Sidechain: duck music when main audio speaks
    # - Safety: normalize final output
    
    # Determine ducking params based on mode
    duck_threshold = "0.02"
    if AUDIO_FORCE_SAFE: duck_threshold = "0.05" # Less sensitive
    
    # Filter Complex:
    # 1. Prepare BG: Loop -> Volume -> Resample to match main
    # 2. Sidechain: Lower BG when Main is active
    # 3. Mix: Combine
    # 4. Loudnorm: Final safety
    
    filter_complex = (
        f"[1:a]volume={volume},aresample=44100[bg];"
        f"[0:a]volume=1.0,aresample=44100[main];"
        f"[bg][main]sidechaincompress=threshold={duck_threshold}:ratio=10:attack=35:release=200[ducked_bg];"
        f"[main][ducked_bg]amix=inputs=2:duration=first:dropout_transition=2[mixed];"
        f"[mixed]{_get_loudnorm_filter()}[out]"
    )
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_video,
        "-stream_loop", "-1", "-i", bg_track,
        "-filter_complex", filter_complex,
        "-map", "0:v", "-map", "[out]",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_video
    ]
    
    if _safe_ffmpeg_run(cmd):
         _audio_meta["effects_used"].append("sidechain_mixing")
         return True

    # Fallback: Simple Mix (No sidechain)
    logger.warning("⚠️ Sidechain mix failed. Applying simple mix...")
    filter_simple = (
        f"[1:a]volume={volume*0.8}[bg];" # Reduce volume further for safety
        f"[0:a]volume=1.0[main];"
        f"[main][bg]amix=inputs=2:duration=first[a];"
        f"[a]{_get_loudnorm_filter()}[out]"
    )
    cmd_fallback = [
        FFMPEG_BIN, "-y", "-i", input_video,
        "-stream_loop", "-1", "-i", bg_track,
        "-filter_complex", filter_simple,
        "-map", "0:v", "-map", "[out]",
        "-c:v", "copy", "-c:a", "aac",
        "-shortest",
        output_video
    ]
    return _safe_ffmpeg_run(cmd_fallback)

def apply_compilation_mastering(input_path: str, output_path: str, original_volume: float = 1.2) -> bool:
    """
    HEAVY REMIX for Compilations.
    Adds 'Stadium' reverb, heavier bass, and compression for a transformative feel.
    
    Args:
        input_path: Input audio/video file
        output_path: Output audio file
        original_volume: Volume multiplier for original audio (default 1.2)
    """
    logger.info(f"🏟️ Applying Compilation Mastering (Heavy Remix): {input_path} (Volume: {original_volume}x)")
    
    # Effects Chain:
    # 1. Bass Boost (Stronger)
    # 2. Exciter (Highs)
    # 3. Stadium Reverb (aecho)
    # 4. Compression (Glue)
    # 5. Limiter
    # 6. Pitch Up + Tempo Fix (Transformative)
    
    filter_chain_list = []
    
    if COMPILATION_MASTER_MODE == "lite":
        filter_chain_list.extend([
            "asetrate=44100*1.03,atempo=1/1.03",
            "equalizer=f=100:t=h:w=120:g=3",
            "acompressor=threshold=-14dB:ratio=2.5",
            f"volume={original_volume}"
        ])
    else:
        # Heavily Distanced (Default) - "Stadium Sound"
        filter_chain_list.extend([
            "asetrate=44100*1.05", # 5% Pitch shift
            "atempo=1/1.05",
            "aecho=0.8:0.88:60:0.4", # Stadium Reverb
            "equalizer=f=60:t=h:w=100:g=5", # Deep Bass
            "equalizer=f=12000:t=h:w=2000:g=3", # Air
            "acompressor=threshold=-12dB:ratio=4:attack=5:release=50",
            f"volume={original_volume}"
        ])
    
    # 3. Loudness Normalization (Mandatory)
    filter_chain_list.append(_get_loudnorm_filter())

    # Add random pitch noise at final stage (Transformative Chaos)
    # Note: rubberband might not be installed, use robust atempo jitter if needed?
    # Keeping existing rubberband logic but wrapping safe.
    # filter_chain += ",rubberband=pitch=random(0.997,1.003)" 
    # Removed rubberband for safety (often missing). Replaced with slight vibrato for "live" feel.
    if not AUDIO_FORCE_SAFE:
         filter_chain_list.append("vibrato=f=4:d=0.1") 
    
    filter_chain = ",".join(filter_chain_list)
    
    cmd = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-af", filter_chain,
        "-vn", 
        "-ac", "2", "-ar", "44100",
        output_path
    ]
    
    if _safe_ffmpeg_run(cmd):
         logger.info("✅ Compilation Mastering Complete")
         return True
    
    # Fallback: Lite Mode
    logger.warning("⚠️ Mastering failed. Trying Lite Mode...")
    lite_chain = f"volume={original_volume},{_get_loudnorm_filter()}"
    cmd_lite = [
        FFMPEG_BIN, "-y", "-i", input_path,
        "-af", lite_chain,
        "-vn", "-ac", "2", "-ar", "44100",
        output_path
    ]
    return _safe_ffmpeg_run(cmd_lite)



def create_continuous_music_mix(output_path: str, target_duration: float, music_dir: str = "music") -> bool:
    """
    Creates a continuous music mix by stitching multiple different songs from the music directory
    until the target duration is reached.
    """
    import random
    import glob
    
    if not os.path.exists(music_dir):
        logger.warning(f"⚠️ Music directory '{music_dir}' not found.")
        return False
        
    music_files = glob.glob(os.path.join(music_dir, "*.mp3")) + glob.glob(os.path.join(music_dir, "*.wav"))
    if not music_files:
        logger.warning(f"⚠️ No music files found in '{music_dir}'.")
        return False
        
    logger.info(f"🎵 Creating continuous mix for {target_duration:.1f}s from {len(music_files)} tracks...")
    
    # Shuffle tracks for randomness
    random.shuffle(music_files)
    
    # Select tracks until we have enough duration
    selected_tracks = []
    current_dur = 0.0
    
    # We might need to loop the playlist if total duration of all songs is less than target
    playlist = music_files.copy()
    
    while current_dur < target_duration:
        if not playlist:
            # Refill playlist if empty (looping the set of songs)
            playlist = music_files.copy()
            random.shuffle(playlist)
            
        track = playlist.pop(0)
        
        # Get track duration
        try:
            cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration", 
                "-of", "default=noprint_wrappers=1:nokey=1", track
            ]
            # Use shell=True on Windows if needed, or just standard run
            # Assuming ffprobe is in path or we use the global FFMPEG_BIN logic if we had access to it here.
            # We'll assume 'ffprobe' is available since this is a helper.
            # Better: use the one defined in this file if possible, but FFMPEG_BIN is defined at top.
            # Let's try to use the FFMPEG_BIN logic from top of file
            ffprobe_bin = FFMPEG_BIN.replace("ffmpeg", "ffprobe") if "ffmpeg" in FFMPEG_BIN.lower() else "ffprobe"
            
            cmd[0] = ffprobe_bin
            dur_str = subprocess.check_output(cmd).decode().strip()
            track_dur = float(dur_str)
            
            if track_dur < 0.5:
                 logger.warning(f"⚠️ Skipping short track: {track} ({track_dur}s)")
                 continue

            selected_tracks.append(track)
            current_dur += track_dur
            
        except Exception as e:
            logger.warning(f"⚠️ Could not read duration of {track}: {e}")
            # If reading fails, skip it
            continue
            
    if not selected_tracks:
        return False
        
    # Create concat list
    temp_list = os.path.join(os.path.dirname(output_path), "mix_list.txt")
    try:
        with open(temp_list, "w") as f:
            for track in selected_tracks:
                f.write(f"file '{os.path.abspath(track).replace(os.sep, '/')}'\n")
        
        # Concat and Trim
        # Concat and Trim
        # -safe 0 is needed for absolute paths
        # 7. Continuous Mix Safety
        # If codecs differ, concat demuxer fails. 
        # Safer approach: Re-encode all to intermediate 44.1k AAC if simple copy fails.
        # OR: Just try copy, if fail, fallback to re-encode concat.
        
        cmd = [
            FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", temp_list,
            "-t", str(target_duration),
            "-c", "copy", # Try copy first for speed
            output_path
        ]
        
        # If copy fails, we re-encode
        if not _safe_ffmpeg_run(cmd):
            logger.info("⚠️ Stream copy failed (mismatched codecs?). Re-encoding mix to unified format...")
            
            # Robust Re-encode Chain
            # - Re-encode to AAC
            # - Resample to 44.1k (Standard)
            # - Loudnorm for the entire mix
            
            cmd_reencode = [
                FFMPEG_BIN, "-y", "-f", "concat", "-safe", "0", "-i", temp_list,
                "-t", str(target_duration),
                "-af", f"aresample=44100,{_get_loudnorm_filter()}",
                "-c:a", "aac",
                "-ab", "192k",
                output_path
            ]
            
            if not _safe_ffmpeg_run(cmd_reencode):
                 logger.error("❌ Audio mix re-encoding failed.")
                 return False
                 
        # Verify output duration and Pad if needed (Tail Protection)
        # (Optional additional check if strict length needed, but -t usually handles it)
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Music mix failed: {e}")
        return False
    finally:
        if os.path.exists(temp_list):
            os.remove(temp_list)

def detect_silence(audio_file: str) -> bool:
    """
    Checks if the audio file has significant silence using ffmpeg silencedetect.
    Returns True if silence is detected (implying speech or sparse audio).
    """
    cmd = [
        FFMPEG_BIN, "-i", audio_file,
        "-af", "silencedetect=noise=-40dB:d=1.5", # Adjusted to 1.5s for speech gaps
        "-f", "null", "-"
    ]
    try:
        # manual run with timeout since we need stderr
        p = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True, encoding='utf-8', errors='replace', timeout=AUDIO_TIMEOUT)
        return "silence_start" in p.stderr
    except Exception as e:
        logger.warning(f"⚠️ Silence detection failed (Timeout/Error): {e}")
        return False # Assume continuous audio if check fails

def build_continuous_segment(output_path: str, segments: list, fade_duration: float = 0.3) -> bool:
    """
    Constructs a single continuous audio file from a list of segments.
    Segments: [{"path": str, "start": float, "duration": float}]
    Handles crossfades between segments if multiple exist.
    """
    if not segments:
        return False
        
    if len(segments) == 1:
        # Simple extraction
        seg = segments[0]
        cmd = [
            FFMPEG_BIN, "-y", 
            "-ss", str(seg['start']),
            "-i", seg['path'],
            "-t", str(seg['duration']),
            "-vn", "-c:a", "libmp3lame", "-q:a", "2",
            output_path
        ]
        return _safe_ffmpeg_run(cmd)

    # Complex Stitching with Crossfades
    # Strategy:
    # 1. Input all files
    # 2. Trim each input to [start, start + duration]
    # 3. Apply concat/crossfade
    # simpler approach: use `acrossfade` filter chain
    
    inputs = []
    filter_chain = []
    
    # We need to assign labels to each trimmed segment first
    for i, seg in enumerate(segments):
        inputs.extend(["-i", seg['path']])
        # Trim: [i:a]atrim=start:end,asetpts=PTS-STARTPTS[a_i]
        # Calculate END point
        end = seg['start'] + seg['duration']
        filter_chain.append(f"[{i}:a]atrim={seg['start']}:{end},asetpts=PTS-STARTPTS[raw_{i}]")

    # Now chain them with acrossfade
    # [raw_0][raw_1]acrossfade=d=0.3[m1];[m1][raw_2]acrossfade=d=0.3[out]
    
    curr = "[raw_0]"
    for i in range(1, len(segments)):
        next_label = f"[raw_{i}]"
        target = f"[mix_{i}]" if i < len(segments) - 1 else "[out]"
        
        # We use acrossfade for smooth transition
        # d=fade_duration
        # curves? default is usually linear/triangular, sufficient.
        filter_chain.append(f"{curr}{next_label}acrossfade=d={fade_duration}{target}")
        curr = target
    
    full_filter = ";".join(filter_chain)
    
    cmd = [
        FFMPEG_BIN, "-y"
    ] + inputs + [
        "-filter_complex", full_filter,
        "-map", "[out]",
        "-vn", "-c:a", "libmp3lame", "-q:a", "2",
        output_path
    ]
    
    return _safe_ffmpeg_run(cmd)
