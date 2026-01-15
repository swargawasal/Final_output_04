
import asyncio
import os
import sys
import logging
import httpx
import json
import time

# Add parent dir to sys.path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

# Setup simplistic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fb_monitor")

from meta_uploader import AsyncMetaUploader

async def monitor_upload(video_path):
    logger.info(f"🚀 Starting Monitor Upload for: {video_path}")
    
    if not os.path.exists(video_path):
        logger.error("❌ File not found!")
        return

    # 1. UPLOAD
    logger.info("📤 Uploading to Facebook...")
    
    # Manually calling internal _upload_to_facebook for control, or public wrapper?
    # Wrapper is easier but might skip if env is wrong. 
    # Let's force it by checking env first.
    if os.getenv("SEND_TO_FACEBOOK", "off").lower() != "on":
        logger.warning("⚠️ SEND_TO_FACEBOOK is OFF in .env. Temporarily mocking it to ON for this script?") 
        # Actually meta_uploader reads os.getenv directly. 
        # We can just call _upload_to_facebook directly as it is static but private-convention.
        # It's better to respect the user's config, assuming they have it ON (they do).
    
    res = await AsyncMetaUploader._upload_to_facebook(video_path, "Debug Upload Monitoring Test #monitoring")
    
    if res.get("status") != "success":
        logger.error(f"❌ Upload Failed Immediately: {res}")
        return

    vid_id = res.get("id")
    page_token = os.getenv("META_PAGE_TOKEN")
    page_id = os.getenv("META_PAGE_ID")
    
    logger.info(f"✅ Upload Accepted. VID_ID: {vid_id}")
    logger.info("🕵️ Starting API Poll Monitor (Every 2s)...")
    logger.info("Press Ctrl+C to stop.")

    # 2. MONITOR LOOP
    url = f"https://graph.facebook.com/v19.0/{vid_id}"
    params = {
        "access_token": page_token,
        "fields": "status,published,errors" 
    }

    async with httpx.AsyncClient() as client:
        while True:
            try:
                resp = await client.get(url, params=params)
                data = resp.json()
                
                # Check for "Object does not exist" error (Deletion)
                if "error" in data:
                    err_code = data['error'].get('code')
                    err_msg = data['error'].get('message')
                    if err_code == 100 and "does not exist" in err_msg:
                        logger.error("❌ VIDEO DELETED BY FACEBOOK (Content Violation Detected)")
                        logger.error(f"Last API Response: {data}")
                        break
                    else:
                         logger.warning(f"⚠️ API Error: {data}")
                else:
                    status = data.get('status', {})
                    phase = status.get('video_status')
                    proc_phase = data.get('processing_phase')
                    
                    print(f"[{time.strftime('%H:%M:%S')}] Status: {phase} | Phase: {proc_phase} | Raw: {json.dumps(data)}")
                    
                    if phase == "error":
                         logger.error("❌ PROCESSING FAILED!")
                         # Try to get more info?
                         break
                         
                    if phase == "ready":
                         logger.info("✅ PROCESSING COMPLETE (Video is Live/Ready)")
                         # We can stop or keep watching
            
            except Exception as e:
                logger.error(f"Poll Error: {e}")
                
            await asyncio.sleep(2)

if __name__ == "__main__":
    # Default file or arg
    target_file = "final_compilations/compile_last_7_Disha_patani_01.mp4"
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        
    asyncio.run(monitor_upload(target_file))
