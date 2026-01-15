import os
import time
import logging
import asyncio
import json
import httpx
from typing import Dict, Optional, Tuple

# Initialize Logger
logger = logging.getLogger("meta_uploader")
logger.setLevel(logging.INFO)

# Constants
GRAPH_API_URL = "https://graph.facebook.com/v19.0"
MAX_RETRIES = 2
RETRY_DELAY = 10
GRAPH_API_URL = "https://graph.facebook.com/v19.0"
MAX_RETRIES = 2
RETRY_DELAY = 10

class AsyncMetaUploader:
    
    @staticmethod
    async def upload_to_meta(video_path: str, caption: str, upload_type: str = "Reels", skip_facebook: bool = False, facebook_caption: str = None) -> Dict:
        """
        Orchestrates uploads to enabled Meta platforms.
        """
        results = {
            "instagram": {"status": "skipped"},
            "facebook": {"status": "skipped"}
        }
        
        if not os.getenv("ENABLE_META_UPLOAD", "no").lower() in ["yes", "true", "on"]:
            logger.info("🚫 Meta Upload Disabled in .env")
            return results

        caption_stripped = caption # logic if needed
        
        # 1. Instagram
        if os.getenv("SEND_TO_INSTAGRAM", "on").lower() in ["yes", "true", "on"]:
            results["instagram"] = await AsyncMetaUploader._upload_to_instagram(video_path, caption)
        else:
             logger.info("🚫 SEND_TO_INSTAGRAM is OFF.")
             results["instagram"] = {"status": "disabled"}
        
        # 2. Facebook (Independent)
        if not skip_facebook and os.getenv("SEND_TO_FACEBOOK", "on").lower() in ["yes", "true", "on"]:
            # Use specific FB caption if provided, else fall back to main caption
            final_fb_caption = facebook_caption if facebook_caption else caption
            results["facebook"] = await AsyncMetaUploader._upload_to_facebook(video_path, final_fb_caption)
        else:
             if skip_facebook:
                  logger.info("🚫 Facebook Skipped (Restricted to Compilation Mode).")
             else:
                  logger.info("🚫 SEND_TO_FACEBOOK is OFF.")
             results["facebook"] = {"status": "disabled/skipped"}
        
        return results

    @staticmethod
    async def _upload_to_instagram(video_path: str, caption: str) -> str:
        ig_id = os.getenv("IG_BUSINESS_ID", "").strip()
        ig_token = os.getenv("IG_BUSINESS_TOKEN", "").strip()
        upload_type_env = os.getenv("META_UPLOAD_TYPE", "Reels").strip().upper()
        
        if not ig_id or not ig_token:
            return {"status": "skipped_no_creds"}
            
        logger.info(f"📸 Starting Instagram Upload ({upload_type_env})...")
        
        url = f"{GRAPH_API_URL}/{ig_id}/media"
        
        try:
            # Step 1: Init (Resumable)
            init_params = {
                "upload_type": "resumable",
                "media_type": "REELS" if upload_type_env == "REELS" else "VIDEO",
                "caption": caption,
                "access_token": ig_token
            }
            
            req_init = await AsyncMetaUploader._retry_request("POST", url, params=init_params)
            if "uri" not in req_init:
                logger.error(f"IG Init Failed: {req_init}")
                return {"status": "failed_init", "error": str(req_init)}
                
            upload_url = req_init["uri"]
            
            # Step 2: Upload Binary
            with open(video_path, "rb") as f:
                data = f.read()

            headers = {
                "Authorization": f"OAuth {ig_token}",
                "offset": "0",
                "Content-Length": str(len(data)),
                "X-Entity-Length": str(len(data)),
                "Content-Type": "video/mp4"
            }
            req_upload = await AsyncMetaUploader._retry_request("POST", upload_url, content=data, headers=headers)
            
            # Extract Container ID
            container_id = None
            if "id" in req_upload:
                container_id = req_upload["id"]
            elif "id" in req_init:
                container_id = req_init["id"]
                
            if not container_id:
                logger.error("IG Upload: No Container ID found.")
                return {"status": "failed_upload", "error": "No Container ID"}
                
            # Wait for processing
            await AsyncMetaUploader._wait_for_media_status(container_id, ig_token)
            
            # Step 4: Publish
            pub_url = f"{GRAPH_API_URL}/{ig_id}/media_publish"
            pub_params = {
                "creation_id": container_id,
                "access_token": ig_token
            }
            
            pub_res = await AsyncMetaUploader._retry_request("POST", pub_url, params=pub_params)
            if "id" in pub_res:
                media_id = pub_res["id"]
                logger.info(f"✅ Instagram Upload Success: {media_id}")
                
                # Fetch Permalink
                link = ""
                try:
                    perm_url = f"{GRAPH_API_URL}/{media_id}"
                    perm_res = await AsyncMetaUploader._retry_request("GET", perm_url, params={"fields": "permalink,shortcode", "access_token": ig_token})
                    link = perm_res.get("permalink") or perm_res.get("shortcode") or ""
                except Exception as e:
                    logger.warning(f"Failed to fetch IG permalink: {e}")

                return {"status": "success", "id": media_id, "link": link}
            else:
                logger.error(f"IG Publish Failed: {pub_res}")
                return {"status": "failed_publish", "error": str(pub_res)}

        except Exception as e:
            logger.error(f"IG Upload Exception: {e}")
            return {"status": "failed", "error": str(e)}

    @staticmethod
    async def _upload_to_facebook(video_path: str, caption: str) -> str:
        page_id = os.getenv("META_PAGE_ID", "").strip()
        page_token = os.getenv("META_PAGE_TOKEN", "").strip()
        # Allow specific override for Facebook, else fallback to global
        upload_type_env = os.getenv("META_UPLOAD_TYPE_FB", os.getenv("META_UPLOAD_TYPE", "Reels")).strip().upper()
        
        if not page_id or not page_token:
            return {"status": "skipped_no_creds"}
            
        logger.info(f"📘 Starting Facebook Upload ({upload_type_env})...")
        
        try:
            endpoint = "video_reels" if upload_type_env == "REELS" else "videos"
            url = f"{GRAPH_API_URL}/{page_id}/{endpoint}"
            
            # 1. Init
            file_size = os.path.getsize(video_path)
            
            init_params = {
                "upload_phase": "start",
                "file_size": file_size,
                "access_token": page_token
            }
            
            req_init = await AsyncMetaUploader._retry_request("POST", url, params=init_params)
            
            if "video_id" not in req_init:
                logger.error(f"FB Init Failed: {req_init}")
                return {"status": "failed_init", "error": str(req_init)}
            
            logger.info(f"FB Init Success: {req_init}")
            # Use video_id for the upload URL (since rupload complains about invalid video id with session_id)
            video_id = req_init["video_id"]
            upload_session_id = req_init.get("upload_session_id", video_id)
            upload_url = f"https://rupload.facebook.com/video-upload/v19.0/{video_id}"
            
            # 2. Upload
            headers = {
                "Authorization": f"OAuth {page_token}",
                "offset": "0",
                "file_offset": "0"
            }
            with open(video_path, "rb") as f:
                data = f.read()
            
            # 2. Upload
            headers = {
                "Authorization": f"OAuth {page_token}",
                "offset": "0",
                "Content-Length": str(len(data)),
                "X-Entity-Length": str(len(data)),
                "Content-Type": "video/mp4"
            }
            req_upload = await AsyncMetaUploader._retry_request("POST", upload_url, content=data, headers=headers)
            
            if "success" not in str(req_upload) and "id" not in req_upload:
                 logger.error(f"FB Chunk Upload Failed: {req_upload}")
                 return {"status": "failed_upload", "error": str(req_upload)}
                 
            # 3. Finish / Publish
            finish_params = {
                "upload_phase": "finish",
                "upload_session_id": upload_session_id,
                "description": caption,
                "access_token": page_token
            }
            
            if upload_type_env == "REELS":
                finish_params["video_state"] = "PUBLISHED"
                finish_params["title"] = caption[:50]
            else:
                # For standard /videos POST, explicitly set published=true
                finish_params["published"] = "true"
            
            req_finish = await AsyncMetaUploader._retry_request("POST", url, params=finish_params)
            
            if "success" in req_finish or "id" in req_finish or "video_id" in req_finish:
                # If finish response has ID, use it. Otherwise fallback to the ID from Init phase
                final_vid_id = req_finish.get('video_id') or req_finish.get('id') or video_id
                logger.info(f"✅ Facebook Upload Success: {final_vid_id}")
                # Construct Link based on type
                if upload_type_env == "REELS":
                     fb_link = f"https://www.facebook.com/reel/{final_vid_id}"
                else:
                     fb_link = f"https://www.facebook.com/watch/?v={final_vid_id}"
                     
                logger.info(f"✅ Facebook Upload Success: {final_vid_id} -> {fb_link}")
                return {"status": "success", "id": final_vid_id, "link": fb_link}
            else:
                logger.error(f"FB Finish Failed: {req_finish}")
                return {"status": "failed_publish", "error": str(req_finish)}
                
        except Exception as e:
            logger.error(f"FB Upload Exception: {e}")
            return {"status": "failed", "error": str(e)}
            
    @staticmethod
    async def _retry_request(method, url, **kwargs) -> Dict:
        """
        Generic retry wrapper for HTTP requests using httpx.
        """
        last_error = None
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            for attempt in range(MAX_RETRIES + 1):
                try:
                    if method == "POST":
                        resp = await client.post(url, **kwargs)
                    else:
                        resp = await client.get(url, **kwargs)
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        # Deep Auth Check
                        if isinstance(data, dict) and 'error' in data:
                            err = data['error']
                            if isinstance(err, dict):
                                e_type = err.get('type', '')
                                e_code = err.get('code')
                                e_sub = err.get('error_subcode')
                                
                                if e_type == 'OAuthException' or e_code == 190 or e_sub == 463:
                                    logger.error("❌ CRITICAL: META AUTH FAILURE. Token expired or invalid.")
                                    logger.info("⚠️ ACTION REQUIRED: Refresh 'IG_BUSINESS_TOKEN' and 'META_PAGE_TOKEN' in .env")
                        
                        return data
                    
                    if 400 <= resp.status_code < 600:
                        try: error_data = resp.json()
                        except: error_data = resp.text
                        
                        logger.warning(f"⚠️ API Error ({resp.status_code}): {error_data}")

                        # Critical Auth Check (Stop Retrying)
                        if isinstance(error_data, dict) and 'error' in error_data:
                            err = error_data['error']
                            if isinstance(err, dict):
                                e_type = err.get('type', '')
                                e_code = err.get('code')
                                e_sub = err.get('error_subcode')
                                # Code 200 = Permission error / User not capable
                                # Code 190 = Access Token Invalid
                                if e_type == 'OAuthException' or e_code in [190, 10, 200] or e_sub == 463:
                                    logger.error("❌ CRITICAL: META AUTH FAILURE. Stopping Retries.")
                                    logger.info("⚠️ ACTION REQUIRED: Check permissions or refresh 'IG_BUSINESS_TOKEN'/'META_PAGE_TOKEN'.")
                                    return error_data
                        
                        if attempt < MAX_RETRIES:
                            logger.info(f"🔄 Retrying in {RETRY_DELAY}s...")
                            await asyncio.sleep(RETRY_DELAY)
                            continue
                        else:
                            # Return error payload on failure
                            return error_data if isinstance(error_data, dict) else {"error": error_data}
                            
                    return resp.json()
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"⚠️ Network Exception: {e}")
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(RETRY_DELAY)
                    else:
                        break
        
        raise last_error

    @staticmethod
    async def _wait_for_media_status(container_id, token, timeout=60):
        start = time.time()
        url = f"{GRAPH_API_URL}/{container_id}"
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.time() - start < timeout:
                try:
                    resp = await client.get(url, params={"access_token": token, "fields": "status_code"})
                    res = resp.json()
                    status = res.get("status_code")
                    if status == "FINISHED":
                        return True
                    if status == "ERROR":
                        return False
                    await asyncio.sleep(3)
                except:
                    await asyncio.sleep(3)
        return False
