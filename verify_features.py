import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.getcwd())

# Mock necessary environment variables
os.environ["FFMPEG_BIN"] = "ffmpeg"
os.environ["FFPROBE_BIN"] = "ffprobe"

# --- MOCKING EXTERNAL LIBS ---
# We must do this BEFORE importing modules that use them
sys.modules["yt_dlp"] = MagicMock()
sys.modules["yt_dlp.utils"] = MagicMock()
sys.modules["cv2"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["PIL"] = MagicMock()
sys.modules["PIL.Image"] = MagicMock()
sys.modules["googleapiclient"] = MagicMock()
sys.modules["googleapiclient.discovery"] = MagicMock()
sys.modules["googleapiclient.http"] = MagicMock() # FIX: Added
sys.modules["googleapiclient.errors"] = MagicMock() # FIX: Added
sys.modules["google.oauth2"] = MagicMock()
sys.modules["google.oauth2.credentials"] = MagicMock()
sys.modules["telegram"] = MagicMock()
sys.modules["telegram.ext"] = MagicMock()
sys.modules["requests"] = MagicMock()
sys.modules["requests.adapters"] = MagicMock()
sys.modules["requests.exceptions"] = MagicMock() # FIX: Added
sys.modules["google_auth_oauthlib"] = MagicMock()
sys.modules["google_auth_oauthlib.flow"] = MagicMock()

# FIX: Complete isolation of Google Auth stack
sys.modules["google"] = MagicMock()
sys.modules["google.auth"] = MagicMock()
sys.modules["google.auth.transport"] = MagicMock()
sys.modules["google.auth.transport.requests"] = MagicMock()

class TestFeatureImplementation(unittest.TestCase):

    def setUp(self):
        # Setup dummy music
        if not os.path.exists("music"):
            os.makedirs("music")
        with open("music/test_track_1.mp3", "w") as f: f.write("dummy")
        with open("music/test_track_2.mp3", "w") as f: f.write("dummy")

    # --- 1. Continuous Music Verification ---
    def test_music_allocation(self):
        print("\n[TEST] Continuous Music Allocation...")
        try:
            from music_manager import ContinuousMusicManager
            
            # Mock get_duration to return fixed values
            ContinuousMusicManager._get_duration = MagicMock(side_effect=lambda x: 60.0) # 60s tracks
            
            manager = ContinuousMusicManager()
            
            # Request 100s music
            segments = manager.allocate_music(100.0)
            
            total_allocated = sum([s['duration'] for s in segments])
            print(f"   -> Needed: 100s, Allocated: {total_allocated}s")
            print(f"   -> Segments: {len(segments)}")
            
            self.assertAlmostEqual(total_allocated, 100.0, delta=0.1)
            self.assertEqual(len(segments), 2) # 60s + 40s
            print("   ✅ PASS")
            
        except ImportError:
            self.fail("Could not import ContinuousMusicManager")
        except Exception as e:
            self.fail(f"Music Allocation Failed: {e}")

    # --- 2. Platform Expansion Verification ---
    def test_downloader_regex(self):
        print("\n[TEST] Downloader Platform Regex...")
        try:
            from downloader import download_video, DownloadIndex
            
            # We explicitly Mock DownloadIndex.find_by_id to intercept the extracted ID
            # This confirms the REGEX worked without needing network/yt-dlp
            
            with patch.object(DownloadIndex, 'find_by_id', side_effect=lambda x: f"/tmp/{x}.mp4") as mock_find:
                
                # Case A: Instagram Reel
                url_ig = "https://www.instagram.com/reel/TEST_IG_ID_123/"
                res = download_video(url_ig)
                # Check if find_by_id was called with expected extraction
                mock_find.assert_any_call("TEST_IG_ID_123")
                print("   ✅ Instagram Reel ID Extracted: TEST_IG_ID_123")
                
                # Case B: Facebook Reel
                url_fb = "https://www.facebook.com/reel/987654321"
                res = download_video(url_fb)
                mock_find.assert_any_call("987654321")
                print("   ✅ Facebook Reel ID Extracted: 987654321")
                
        except ImportError:
             self.fail("Could not import downloader modules")
        except Exception as e:
             self.fail(f"Regex Parsing Failed: {e}")

    # --- 3. Branding Logic Verification ---
    def test_branding_overlay(self):
        print("\n[TEST] Branding Overlay Command Construction...")
        try:
            from text_overlay import add_logo_overlay
            
            # Mock subprocess.run AND os.path.exists
            # We need to mock os.path.exists specifically for the logic checks
            original_exists = os.path.exists
            
            def side_effect_exists(path):
                if "logo.png" in str(path) or "input.mp4" in str(path):
                    return True
                return original_exists(path)

            with patch("subprocess.run") as mock_run, \
                 patch("time.time", return_value=12345), \
                 patch("os.path.exists", side_effect=side_effect_exists):
                
                mock_run.return_value.returncode = 0
                
                # Test Logo
                add_logo_overlay("input.mp4", "output.mp4", "logo.png", lane_context="caption")
                
                # Check args
                if mock_run.call_args:
                    args = mock_run.call_args[0][0]
                    # args is list of cmd parts. Join to search.
                    cmd_str = " ".join(args)
                    
                    # Verify collision avoidance match
                    if "h-h-50" in cmd_str:
                        print(f"   ✅ Logo Collision Avoidance Detected (h-h-50)")
                    else:
                        print(f"   ⚠️ CMD: {cmd_str}")
                        self.fail(f"Logo collision avoidance missing")
                else:
                    self.fail("subprocess.run was NOT called (Method exited early?)")
                    
        except ImportError:
            self.fail("Could not import text_overlay")
            
    # --- 4. Error Handling Verification ---
    def test_quota_lock(self):
        print("\n[TEST] Quota Lock Logic...")
        lock_file = "youtube_quota.lock"
        if os.path.exists(lock_file): os.remove(lock_file)
        
        try:
            # Re-import to handle mocked modules
            # Now these functions exist!
            from uploader import check_quota_lock, set_quota_lock
            
            # Verify clean state
            self.assertFalse(check_quota_lock())
            
            # Set lock
            set_quota_lock()
            
            # Verify locked state
            self.assertTrue(check_quota_lock())
            print("   ✅ Lock mechanism functional")
            
            # Cleanup
            if os.path.exists(lock_file): os.remove(lock_file)
            
        except ImportError as e:
            self.fail(f"Could not import uploader functions: {e}")

    # --- 5. Ferrari Composer Syntax Verification ---
    def test_ferrari_command(self):
        print("\n[TEST] Ferrari Composer Command Syntax...")
        try:
            # Import compiler after mocking
            import compiler
            from compiler import apply_ferrari_composer
            
            # Additional mocks for compiler dependencies if needed
            # (handled by module-level sys.modules)

            # We need to mock _run_command or subprocess.run
            # compiler.py uses _run_command which wraps subprocess.run
            # Let's patch _run_command in compiler module
            
            with patch("compiler._run_command") as mock_run, \
                 patch("compiler._get_video_info", return_value={'duration': 100.0}), \
                 patch("compiler._has_audio_stream", return_value=True), \
                 patch("compiler._get_ffmpeg_encoder", return_value="libx264"):
                
                mock_run.return_value = True
                
                input_file = "test_input.mp4"
                output_file = "test_output.mp4"
                
                # Mock file existence
                with patch("os.path.exists", return_value=True):
                    apply_ferrari_composer(
                        input_file, output_file,
                        color_intensity=0.5,
                        filter_type="cinematic"
                    )
                
                # Verify Call
                if mock_run.called:
                    args = mock_run.call_args[0][0] # First arg is cmd list
                    cmd_str = " ".join(args)
                    
                    # Check 1: Output Path is Last
                    self.assertEqual(args[-1], output_file, "Output path must be the LAST argument")
                    print("   ✅ Output Path is Last")
                    
                    # Check 2: Float Formatting
                    # Look for rs=... it should be like rs=-0.015
                    # If it's rs=-0.015000000000000001 failure
                    
                    import re
                    # Extract rs value
                    match = re.search(r"rs=([-0-9\.]+)", cmd_str)
                    if match:
                        val = match.group(1)
                        if len(val) > 7: # -0.123 is 6 chars. -0.123456... is longer
                             print(f"   ⚠️ Warning: Long float detected? {val}")
                             # It might be okay if it's strictly 3 decimals but regex captures more?
                             # Logic check:
                             # We expect .3f so max 3 decimals.
                             if len(val.split('.')[-1]) > 3:
                                 self.fail(f"Float precision too high: {val}")
                        print(f"   ✅ Float Precision OK ({val})")
                    else:
                        print("   ⚠️ Could not find 'rs=' param to check.")

                    # Check 3: Agate Removal (Fix for EINVAL)
                    if "agate" in cmd_str:
                        self.fail("❌ 'agate' filter found! It should have been removed.")
                    print("   ✅ 'agate' filter verified ABSENT")

                    # Check 4: Pan Syntax Fix (< vs =)
                    if "pan=stereo" in cmd_str:
                        if "<" in cmd_str.split("pan=stereo")[1].split("]")[0]:
                             self.fail("❌ Invalid 'pan' syntax found ('<'). Should be '='.")
                        if "c0=c0" not in cmd_str:
                             self.fail("❌ Correct 'pan' syntax ('c0=c0') not found.")
                        print("   ✅ 'pan' syntax verified (Using '=')")

                    print(f"   ✅ Command Syntax Validated")
                else:
                    self.fail("apply_ferrari_composer did not call _run_command")

        except ImportError as e:
            self.fail(f"Could not import compiler: {e}")
        except Exception as e:
            self.fail(f"Ferrari Test Failed: {e}")

if __name__ == '__main__':
    unittest.main()
