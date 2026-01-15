import os
import asyncio
import logging
import httpx
from dotenv import load_dotenv

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("status_check")

VID_ID = "1274012037885988"
PAGE_TOKEN = os.getenv("META_PAGE_TOKEN")

async def check_status():
    async with httpx.AsyncClient() as client:
        print(f"--- CHECKING VIDEO: {VID_ID} ---")
        
        # 1. Check Page Status
        page_url = f"https://graph.facebook.com/v19.0/{os.getenv('META_PAGE_ID')}"
        page_resp = await client.get(page_url, params={"access_token": PAGE_TOKEN, "fields": "name,is_published,link"})
        print(f"Page Info: {page_resp.json()}")
        
        # 2. Check Video Status
        url = f"https://graph.facebook.com/v19.0/{VID_ID}"
        params = {
            "access_token": PAGE_TOKEN,
            "fields": "status,published,permalink_url,post_views,title,privacy,description,copyright_monitoring_status"
        }
        
        resp = await client.get(url, params=params)
        data = resp.json()
        print(f"\n--- VIDEO STATUS ---")
        print(data)
        
        if "error" in data:
            print("\n❌ API Error: Video might be deleted or ID is invalid.")

if __name__ == "__main__":
    asyncio.run(check_status())
