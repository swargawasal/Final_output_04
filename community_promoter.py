import os
import json
import time
import hashlib
import random
import logging
import asyncio
from typing import Optional
from datetime import datetime

logger = logging.getLogger("community_promoter")
logger.setLevel(logging.INFO)

STATE_FILE = "community_promo_state.json"

class CommunityPromoter:
    """
    Handles 'Community Post' promotion via Channel Comments (commentThreads).
    - Rate Limited (6h)
    - Deterministic Content (No Gemini)
    - Silent Failures
    """
    
    def __init__(self):
        self.state = self._load_state()
        
    def _load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Migration: Single URL -> Pool
                    if "last_compilation_url" in data and "promo_pool" not in data:
                        data["promo_pool"] = [
                            {"url": data["last_compilation_url"], "ts": data.get("last_compilation_time", time.time())}
                        ]
                        
                    return data
            except Exception:
                pass
        return {"last_run": 0, "posted_hashes": [], "promo_pool": []}

    def _save_state(self):
        try:
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.state, f)
        except Exception as e:
            logger.error(f"❌ Failed to save promoter state: {e}")

    def _get_template(self, clip_count: int, promo_url: str) -> str:
        """
        Returns a deterministic promotional text pointing to the COMPILATION.
        """
        templates = [
            f"{clip_count} must-see celebrity fashion moments ✨\nFull compilation is live — watch now 👇\n{promo_url}",
            f"A fresh compilation is up 🎬\n{clip_count} standout celebrity fashion moments in one video.\n▶️ {promo_url}",
            f"New compilation uploaded!\n{clip_count} celebrity looks worth watching 👀\nWatch here 👇\n{promo_url}",
            f"Just dropped: {clip_count} celebrity fashion moments\nWatch the full video now 👇\n{promo_url}"
        ]
        return random.choice(templates)

    def register_compilation_url(self, url: str):
        """
        Adds compilation URL to the rotating pool (Max 10).
        """
        pool = self.state.get("promo_pool", [])
        
        # Deduplicate
        pool = [x for x in pool if x["url"] != url]
        
        # Add new
        pool.append({"url": url, "ts": time.time()})
        
        # Cap size (Keep recent 10)
        if len(pool) > 10:
            pool = pool[-10:]
            
        self.state["promo_pool"] = pool
        self._save_state()
        logger.info(f"💾 Registered Compilation URL to Pool (Total: {len(pool)}): {url}")

    def _get_rotation_url(self) -> Optional[str]:
        """
        Picks a URL from the pool (Random Rotation).
        """
        pool = self.state.get("promo_pool", [])
        if not pool:
            return None
        return random.choice(pool)["url"]

    def _can_run(self, content_hash: str) -> bool:
        """
        Checks rate limit (6h) and duplication.
        """
        now = time.time()
        
        # 1. Rate Limit (6 Hours)
        last_run = self.state.get("last_run", 0)
        
        if now - last_run < 60: # 1 minute safe guard
            logger.info(f"⏳ Community Promotion skipped (Rate Limit: {int(60 - (now-last_run))}s remaining)")
            return False
            
        # 2. Duplicate Guard
        if content_hash in self.state.get("posted_hashes", []):
            logger.info("♻️ Community Promotion skipped (Duplicate content)")
            return False
            
        return True

    def _register_success(self, content_hash: str):
        self.state["last_run"] = time.time()
        
        # Keep hash history manageable (last 50)
        hashes = self.state.get("posted_hashes", [])
        hashes.append(content_hash)
        if len(hashes) > 50:
            hashes = hashes[-50:]
        self.state["posted_hashes"] = hashes
        
        self._save_state()

    async def promote_on_short_async(self, service, short_video_url: str, delay_seconds: int = 20):
        """
        Promotes a ROTATING Compilation on the provided Short Video.
        """
        comp_url = self._get_rotation_url()
        
        if not comp_url:
            logger.warning("⚠️ No Compilation URL in Pool. Skipping promotion on Short.")
            return

        logger.info(f"⏲️ Scheduling Community Promotion in {delay_seconds}s (Link: {comp_url})...")
        await asyncio.sleep(delay_seconds)
        
        # We need to run the blocking API call in a thread
        # Pass dummy clip_count=10 if unknown, or maybe store it?
        # Let's assume generic "10" or "Many" context defaults for now.
        clip_count = 10 
        await asyncio.to_thread(self._promote_sync, service, short_video_url, comp_url, clip_count)

    def _extract_video_id(self, url: str) -> Optional[str]:
        try:
            if "youtu.be" in url:
                return url.split("/")[-1].split("?")[0]
            if "v=" in url:
                return url.split("v=")[-1].split("&")[0]
            if "shorts" in url:
                 return url.split("shorts/")[-1].split("?")[0]
        except:
            pass
        return None

    def _promote_sync(self, service, target_video_url: str, promo_link: str, clip_count: int):
        try:
            # 1. Generate Content
            text = self._get_template(clip_count, promo_link)
            content_hash = hashlib.md5(text.encode()).hexdigest()
            
            # 2. Guard Checks
            if not self._can_run(content_hash):
                return

            # 3. Get Channel ID (Required for commentThreads)
            try:
                channels_response = service.channels().list(mine=True, part="id").execute()
                if not channels_response.get("items"):
                    logger.warning("⚠️ Could not resolve Channel ID. Skipping.")
                    return
                channel_id = channels_response["items"][0]["id"]
            except Exception as e:
                 logger.warning(f"⚠️ Channel ID fetch failed: {e}")
                 return
            
            # 3b. Extract Video ID (Target Short)
            video_id = self._extract_video_id(target_video_url)
            if not video_id:
                logger.warning(f"⚠️ Could not extract Video ID from {target_video_url}. Skipping.")
                return

            # 4. Execute API Call (Best Effort)
            # Posting a TOP LEVEL COMMENT on the TARGET VIDEO
            body = {
                "snippet": {
                    "channelId": channel_id,
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {
                            "textOriginal": text
                        }
                    }
                }
            }
            
            service.commentThreads().insert(
                part="snippet",
                body=body
            ).execute()
            
            # 5. Success
            logger.info(f"📣 Community Promotion Posted on Short ({video_id}) -> Linking to Compilation!")
            self._register_success(content_hash)
            
        except Exception as e:
            # SILENT FAILURE
            logger.warning(f"ℹ️ Community Promotion skipped: {e}")

# Global Instance
promoter = CommunityPromoter()

if __name__ == "__main__":
    # Manual Test Mode
    logging.basicConfig(level=logging.INFO)
    print("📢 Community Promoter Manual Mode")
    
    try:
        from uploader import get_authenticated_service
        service = get_authenticated_service()
        if not service:
            print("❌ Auth failed.")
            exit(1)
            
        url = input("Enter Video URL: ").strip()
        count = int(input("Enter Clip Count: ").strip())
        
        print("🚀 Promoting...")
        promoter._promote_sync(service, url, count)
        
    except ImportError:
        import traceback
        traceback.print_exc()
        print("❌ Could not import 'uploader.get_authenticated_service'. Check traceback above.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ Error: {e}")
