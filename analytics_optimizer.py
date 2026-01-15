
import os
import json
import logging
import datetime
from typing import Optional, Dict, Any
from googleapiclient.discovery import build
import google.generativeai as genai
from uploader import get_valid_credentials

logger = logging.getLogger("analytics_optimizer")

# Configuration
CACHE_FILE = "analytics_cache.json"
CACHE_DURATION_DAYS = 30

class AnalyticsOptimizer:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found. Optimization will be disabled.")
            self.gemini_available = False
        else:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.gemini_available = True
            
        self.cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Any]:
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_cache(self, data: Dict[str, Any]):
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ Failed to save analytics cache: {e}")

    def get_analytics_service(self):
        """Builds the YouTube Analytics API service."""
        try:
            creds = get_valid_credentials()
            if not creds:
                logger.error("❌ No valid credentials available for Analytics.")
                return None
            return build("youtubeAnalytics", "v2", credentials=creds)
        except Exception as e:
            logger.error(f"❌ Failed to build Analytics service: {e}")
            return None

    def fetch_viewer_data(self) -> Optional[str]:
        """
        Fetches channel view statistics organized by day of week and hour.
        Returns a formatted string summary of the data.
        """
        service = self.get_analytics_service()
        if not service:
            return None

        try:
            # Query last 90 days for better statistical significance
            end_date = datetime.date.today().strftime("%Y-%m-%d")
            start_date = (datetime.date.today() - datetime.timedelta(days=90)).strftime("%Y-%m-%d")

            logger.info(f"📊 Fetching YouTube Analytics data from {start_date} to {end_date}...")

            # We want to know when people are watching: simple views metric
            # Dimensions: dayOfWeek, hour
            request = service.reports().query(
                ids="channel==MINE",
                startDate=start_date,
                endDate=end_date,
                metrics="views",
                dimensions="dayOfWeek,hour",
                sort="-views",
                maxResults=25  # Top 25 slots is enough to find the peak
            )
            response = request.execute()

            rows = response.get("rows", [])
            if not rows:
                logger.warning("⚠️ No analytics data found (new channel?).")
                return None
            
            # Format for Gemini
            # Row format: [dayOfWeek (0=Mon), hour (0-23), views]
            # Map day index to name
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            
            summary = "Recent Viewership Data (Top Times):\n"
            for row in rows:
                day_idx = int(row[0])
                hour = int(row[1])
                views = int(row[2])
                day_name = days[day_idx] if 0 <= day_idx < 7 else "Unknown"
                summary += f"- {day_name} at {hour}:00 -> {views} views\n"
            
            return summary

        except Exception as e:
            logger.error(f"❌ Analytics API Error: {e}")
            return None

    def analyze_with_gemini(self, analytics_summary: str) -> Optional[Dict[str, Any]]:
        """
        Asks Gemini to pick the best time provided the analytics data.
        """
        if not self.gemini_available:
            return None

        prompt = (
            "You are a YouTube Growth Strategist. Analyze the following viewership data to determine the optimal upload time.\n"
            "STRATEGY: The best time to upload is exactly 1 HOUR BEFORE the peak viewing time to allow indexing.\n\n"
            f"DATA:\n{analytics_summary}\n\n"
            "TASK: Return a JSON object with the recommended upload day and time (24h format).\n"
            "FORMAT: { \"day\": \"Monday\", \"hour\": 14, \"reason\": \"Peak is at 15:00, so upload at 14:00.\" }\n"
            "Return ONLY the JSON."
        )

        try:
            logger.info("🧠 Sending Analytics data to Gemini for optimization...")
            response = self.model.generate_content(prompt)
            text = response.text.strip().replace("```json", "").replace("```", "")
            data = json.loads(text)
            return data
        except Exception as e:
            logger.error(f"❌ Gemini Analysis Failed: {e}")
            return None

    def get_optimal_upload_time(self) -> Optional[Dict[str, Any]]:
        """
        Main method to get the optimization result.
        Checks cache first. If expired/missing, fetches fresh data.
        """
        now = datetime.datetime.utcnow().timestamp()
        
        # 1. Check Cache
        cached_result = self.cache.get("optimization_result")
        last_fetch = self.cache.get("last_fetch_timestamp", 0)
        
        is_expired = (now - last_fetch) > (CACHE_DURATION_DAYS * 86400)
        
        if cached_result and not is_expired:
            logger.info(f"✨ Using Cached Upload Optimization (Expires in {int((CACHE_DURATION_DAYS * 86400 - (now - last_fetch))/86400)} days)")
            return cached_result
            
        # 2. Fetch Fresh Data
        logger.info("🔄 Cache expired or missing. Running full optimization cycle...")
        
        raw_data = self.fetch_viewer_data()
        if not raw_data:
            return cached_result # Fallback to old cache if fetch fails
            
        result = self.analyze_with_gemini(raw_data)
        
        if result:
            self.cache["optimization_result"] = result
            self.cache["last_fetch_timestamp"] = now
            self._save_cache(self.cache)
            logger.info(f"✅ New Optimization Saved: {result}")
            return result
        
        return cached_result

    def calculate_next_publish_time(self, day_name: str, hour: int) -> Optional[str]:
        """
        Calculates the next ISO 8601 UTC timestamp for the given day and hour.
        Assumes the input day/hour refers to YouTube Analytics default timezone (PST/PDT).
        """
        try:
            days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            target_weekday = days.index(day_name.capitalize())
            
            # Use UTC now as base
            now_utc = datetime.datetime.utcnow()
            
            # Estimate PST offset (approx -8 for standard, -7 for daylight)
            # For simplicity & safety, we'll assume -7 to be safe (earlier in UTC) or -8.
            # A fixed offset of -8 means 14:00 PST = 22:00 UTC. 
            # If we are wrong by 1 hour (DST), it's fine for YouTube upload optimization.
            pst_offset = -8
            
            # Start with current UTC time
            # We need to find a UTC time where (UTC_time + pst_offset) has the target weekday and hour.
            # So: target_utc_hour = hour - pst_offset
            
            target_utc_hour = hour - pst_offset
            if target_utc_hour >= 24:
                target_utc_hour -= 24
                # Moves to next day relative to PST inputs
                # But weekday calculation needs care.
                pass 
                
            # Simpler approach:
            # 1. Create a naive datetime for "Next [Day] at [Hour]"
            # 2. Treat it as PST.
            # 3. Convert to UTC.
            
            today = datetime.datetime.utcnow().date() # Close enough to PST date usually
            
            # Find next occurrence of weekday
            days_ahead = target_weekday - today.weekday()
            if days_ahead <= 0: # Target day already happened this week or is today
                days_ahead += 7
            
            # However, if it is today, check if hour passed.
            # Since we switched to next week if <= 0, we miss Today's later slots.
            # Fix:
            days_ahead = target_weekday - today.weekday()
            if days_ahead < 0:
                days_ahead += 7
            
            target_date = today + datetime.timedelta(days=days_ahead)
            
            # Construct target PST time
            # Note: This is naive. Ideally use pytz but environment might not have it.
            # We assume PST is UTC-8.
            
            target_pst_dt = datetime.datetime(
                target_date.year, target_date.month, target_date.day,
                hour, 0, 0
            )
            
            # Convert to UTC (-8h reverse is +8h)
            target_utc_dt = target_pst_dt + datetime.timedelta(hours=8)
            
            # Ensure it is in the future relative to NOW
            if target_utc_dt < now_utc:
                target_utc_dt += datetime.timedelta(days=7)
                
            return target_utc_dt.isoformat() + "Z"
            
        except Exception as e:
            logger.error(f"❌ Date calculation failed: {e}")
            return None

# Singleton instance
optimizer = AnalyticsOptimizer()
