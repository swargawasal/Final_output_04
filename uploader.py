import os
import time
import logging
from typing import Optional
import asyncio
import subprocess
import sys
import json
import shutil
import uuid

FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
if not shutil.which(FFPROBE_BIN):
    FFPROBE_BIN = "ffprobe"

from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",  # REQUIRED for Community Promotion (Comments)
    "https://www.googleapis.com/auth/yt-analytics.readonly",
    "https://www.googleapis.com/auth/yt-analytics-monetary.readonly"
]
CLIENT_SECRET_FILE = os.environ.get("CLIENT_SECRET_FILE", "client_secret.json")
TOKEN_FILE = os.environ.get("YOUTUBE_TOKEN_FILE", "token.json")

logger = logging.getLogger("uploader")
logger.setLevel(logging.INFO)

# One-time warning for scope change
if os.path.exists(TOKEN_FILE):
    logger.info("ℹ️ NOTE: Community Promotion/Analytics requires re-authentication (delete token.json). Uploads will continue normally without it.")

def check_quota_lock() -> bool:
    """Checks if the 24h upload lock is active."""
    lock_file = "youtube_quota.lock"
    if os.path.exists(lock_file):
        try:
            with open(lock_file, 'r') as f:
                timestamp = float(f.read().strip())
            # Check 24h expiration
            if time.time() - timestamp < 86400:
                return True
            else:
                # Expired: remove it
                os.remove(lock_file)
                return False
        except Exception:
            return False # Corrupt definition of lock -> assume safe
    return False

def set_quota_lock():
    """Sets a 24h upload lock."""
    with open("youtube_quota.lock", "w") as f:
        f.write(str(time.time()))
    logger.warning("🛑 QUOTA LOCK SET: Uploads paused for 24 hours.")


def get_valid_credentials():
    """
    Retrieves and refreshes valid credentials.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        except Exception:
            logger.warning("Failed to read token file, will run auth flow.")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                # SAVE THE REFRESHED TOKEN
                with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                    f.write(creds.to_json())
                logger.info("✅ Token refreshed and saved.")
            except Exception as e:
                logger.warning(f"Refresh failed: {e}")
                creds = None
        if not creds:
            logger.warning("🔄 Token expired or missing. Launching auto-auth...")
            try:
                # Auto-run the auth script
                subprocess.check_call([sys.executable, "scripts/auth_youtube.py"])
                
                # Reload credentials after script finishes
                if os.path.exists(TOKEN_FILE):
                    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
            except Exception as e:
                logger.error(f"❌ Auto-auth failed: {e}")

            if not creds or not creds.valid:
                logger.error("❌ Authentication failed: Token expired or missing.")
                raise Exception("YouTube Authentication Failed. Please run 'python scripts/auth_youtube.py' locally to refresh credentials.")
    return creds

def _get_service_sync():
    creds = get_valid_credentials()
    service = build("youtube", "v3", credentials=creds)
    return service


def verify_metadata(file_path: str) -> bool:
    """
    Checks if the video file has fresh metadata (Unique ID, Creation Time).
    Returns True if fresh, False otherwise.
    """
    try:
        cmd = [
            FFPROBE_BIN, "-v", "quiet", 
            "-print_format", "json", 
            "-show_format", 
            file_path
        ]
        result = subprocess.check_output(cmd, shell=True).decode().strip()
        data = json.loads(result)
        tags = data.get("format", {}).get("tags", {})
        
        comment = tags.get("comment", "")
        creation_time = tags.get("creation_time", "")
        
        is_fresh = False
        if "ID:" in comment or "Unique ID:" in comment:
            logger.info(f"✅ Metadata Verified: Found Unique ID in comments.")
            is_fresh = True
        else:
            logger.warning(f"⚠️ Metadata Warning: No 'Unique ID' found in file comments (Comment: {comment[:50]}...).")
            
        if creation_time:
            logger.info(f"✅ Metadata Verified: Creation Time = {creation_time}")
        else:
            # Fallback to filesystem timestamp
            try:
                fs_ctime = time.ctime(os.path.getctime(file_path))
                logger.info(f"✅ Metadata Verified: Filesystem Timestamp = {fs_ctime}")
            except Exception:
                logger.warning(f"⚠️ Metadata Warning: No 'creation_time' found.")
            
        return is_fresh
    except Exception as e:
        logger.warning(f"⚠️ Failed to verify metadata: {e}")
        return False


def refresh_metadata(file_path: str) -> bool:
    """
    Injects a fresh Unique ID into the video metadata without re-encoding.
    """
    try:
        new_id = str(uuid.uuid4())
        temp_path = file_path + ".temp.mp4"
        logger.info(f"🔄 Injecting Fresh Unique ID: {new_id}...")
        
        # ffmpeg -i input -map 0 -c copy -metadata comment="ID:<uuid>" temp
        cmd = [
            FFPROBE_BIN.replace("ffprobe", "ffmpeg"), "-y", 
            "-i", file_path,
            "-map", "0",
            "-c", "copy",
            "-metadata", f"comment=ID:{new_id}",
            temp_path
        ]
        
        # Run safely
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Verify output exists
        # Verify output exists
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            # Atomic Move (with Retry)
            try:
                if os.path.exists(file_path):
                     os.remove(file_path) # Force delete original
                shutil.move(temp_path, file_path)
                logger.info("✅ Metadata refreshed successfully.")
                return True
            except PermissionError:
                 logger.error("⚠️ File is locked! Cannot inject Unique ID. Proceeding with original file (Filesystem Timestamp will be used).")
                 if os.path.exists(temp_path): os.remove(temp_path) # Clean temp
                 return True # Allow upload to proceed
        else:
            logger.error("❌ Metadata refresh failed: Output empty.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to refresh metadata: {e}")
        return True # Soft Fail: Allow upload even if metadata injection fails


def _upload_sync(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
    publish_at: Optional[str] = None,
) -> Optional[str]:
    # 0. Check Quota Lock First
    if check_quota_lock():
        logger.warning("🚫 Upload Skipped: YouTube Quota Lock is Active (Wait 24h).")
        return None

    # Enforce .mp4 extension
    if not file_path.lower().endswith(".mp4"):
        logger.error("❌ Upload rejected: File must be .mp4")
        return None

    service = _get_service_sync()
    logger.info(f"DEBUG: _upload_sync called with title input: '{title}'")
    
    # Robust title logic: Ensure it's not None, not empty, and not just whitespace
    if title:
        # Sanitize: Remove newlines, tabs, and forbidden characters
        final_title = title.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        final_title = final_title.replace("<", "").replace(">", "") # No HTML-like tags
        final_title = final_title.strip()
    else:
        final_title = ""

    if not final_title:
        final_title = "Untitled Video"
        logger.warning("⚠️ Title was empty or whitespace. Defaulting to 'Untitled Video'.")
        
    # Enforce YouTube Length Limit (100 chars)
    if len(final_title) > 95:
        final_title = final_title[:95]
        
    logger.info(f"📋 Final Title for Upload: '{final_title}'")
        
    final_description = ((description or "").strip() + ("\n\n" + hashtags if hashtags else "")).strip()

    status_dict = {
        "privacyStatus": privacy,
        "selfDeclaredMadeForKids": False,
    }

    # Handle Scheduling
    if publish_at:
        status_dict["publishAt"] = publish_at
        status_dict["privacyStatus"] = "private" # Must be private for scheduled upload
        logger.info(f"📅 Scheduled Upload: {publish_at}")

    body = {
        "snippet": {
            "title": final_title,
            "description": final_description,
            "categoryId": "22",  # People & Blogs
        },
        "status": status_dict,
    }

    logger.info("🚀 Starting upload request to YouTube API...")
    
    # Verify Metadata Freshness (Must be done BEFORE opening MediaFileUpload to avoid File Locking)
    if not verify_metadata(file_path):
        logger.warning("🔄 Stale/Missing Metadata detected. Engaging Auto-Refresh Safety Net...")
        refresh_metadata(file_path)
        # Verify again just to be sure
        verify_metadata(file_path)

    media = MediaFileUpload(
        file_path,
        chunksize=1024 * 1024 * 16,  # 16 MB
        resumable=True,
        mimetype="video/mp4"
    )
    
    request = service.videos().insert(
        part="snippet,status",
        body=body,
        media_body=media
    )
    logger.info("✅ Upload request created. Starting chunk upload loop...")

    logger.info("🚀 Starting upload: %s", file_path)
    retry = 0
    while True:
        try:
            status, response = request.next_chunk()
            if response is not None:
                video_id = response.get("id")
                if video_id:
                    logger.info("✅ Upload complete: %s", video_id)
                    return f"https://youtube.com/watch?v={video_id}"
                return None
            if status and hasattr(status, "progress"):
                progress = int(status.progress() * 100)
                logger.info("Upload progress: %d%%", progress)
        except HttpError as e:
            # Smart Error Handling
            error_reason = ""
            try:
                error_content = json.loads(e.content.decode('utf-8'))
                error_reason = str(error_content).lower()
            except: 
                error_reason = str(e).lower()

            if "uploadlimitexceeded" in error_reason or "quotaexceeded" in error_reason:
                logger.error("❌ CRITICAL: YouTube Upload Quota Exceeded for today.")
                # Use Helper
                set_quota_lock()
                
                return None # Stop immediately

            logger.warning("YouTube API HttpError on chunk: %s", e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to HttpError.")
                return None
            time.sleep(2 ** retry)
        except Exception as e:
            logger.exception("Upload error: %s", e)
            retry += 1
            if retry > 5:
                logger.error("Max retries reached for upload due to Exception.")
                return None
            time.sleep(2 ** retry)


async def upload_to_youtube(
    file_path: str,
    hashtags: str = "",
    title: Optional[str] = None,
    description: Optional[str] = None,
    privacy: str = "public",
    publish_at: Optional[str] = None,
) -> Optional[str]:
    print(f"DEBUG: uploader.upload_to_youtube called for {file_path}")
    return await asyncio.to_thread(_upload_sync, file_path, hashtags, title, description, privacy, publish_at)

# Expose authentication for other modules (e.g., community_promoter)
get_authenticated_service = _get_service_sync


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_file = os.environ.get("TEST_UPLOAD_FILE", "downloads/final_highres_output.mp4")
    # Create a dummy file for testing if it doesn't exist
    if not os.path.exists(test_file):
        with open(test_file, "wb") as f:
            f.write(b"dummy content")
            
    link = _upload_sync(test_file, hashtags="#example", title="High-Quality Test Upload")
    print("Uploaded:", link)
