"""
Monetization Brain Module (Gemini Authority Mode)
-------------------------------------------------
Acts as the YOUTUBE SHORTS CAPTION EDITOR & SAFETY OFFICER.
Goal: Pass YPP Human Review by enforcing strictly editorial/transformative captions.

**SINGLE SOURCE OF TRUTH: GEMINI**
- No OpenAI usage.
- Strict Text Parsing for robustness.
"""

import os
import json
import logging
import re
import google.generativeai as genai
from typing import Dict, Optional, List
from datetime import datetime
import shutil
import tempfile

logger = logging.getLogger("monetization_brain")

# YPP STRICT EDITOR PROMPT (GEMINI AUTHORITY)
# YPP STRICT EDITOR PROMPT (GEMINI AUTHORITY)
EDITOR_PROMPT = """
YOU ARE THE MONETIZATION & CAPTION AUTHORITY.

GLOBAL RULES (HARD):
- GEMINI IS THE ONLY MODEL USED.
- NEVER output labels like: "Aesthetic", "Editorial", "Safe", "Approved".
- NEVER overwrite captions with classification words.
- ONE caption = ONE truth.

CAPTION GENERATION RULES
You must generate a DISPLAY-READY caption.
- Natural, human, editorial tone
- Monetization safe (YPP friendly)
- No sexual wording
- No thirst language
- Emojis allowed (max 1)
- Length: 8-15 words ONLY
- Do NOT include hashtags
- Do NOT include usernames
- Do NOT include formatting symbols (*, #, [], etc.)

Examples of GOOD captions:
- "Mixing vintage denim with modern confidence for a timeless look"
- "A quiet moment of reflection capturing the essence of style"
- "Every detail feels intentional making this outfit truly stand out"
- "Effortless energy that defines the modern aesthetic perfectly"

RETURN FORMAT (CRITICAL)
YOUTUBE POLICY KNOWLEDGE (OFFICIAL -"REUSED CONTENT"):
- **Prohibited**: "Short videos you compiled from other social media websites... without significant original commentary."
- **Allowed (Transformative)**: "Clips stitched together where you provide **critical commentary**." or "Shorts that you've edited to add **voiceover** that changes the **meaning** of the source."

ANALYSIS INSTRUCTIONS (STRICT RUBRIC):
1. **TIER 1 (Policy Safe / Monetizable) -> Score 90-100**:
   - **Requirement**: MUST have 'Voiceover' that changes meaning OR adds educational value.
   - **PLUS**: 'Inpainting' (Logo Removal) / Clean Visuals.
   - *Verdict*: "Transformative (Commentary Added)"

2. **TIER 2 (Borderline Policy) -> Score 50-89**:
   - Has 'Voiceover' but it only describes the video (Narrative, but low critical value).
   - OR Has 'Inpainting' but no Voiceover (Visual transformation only).
   - *Verdict*: "Transformative (Weak)"

3. **TIER 3 (Policy Violation Risk) -> Score 0-49**:
   - No Voiceover. No Logo Removal.
   - Only Speed/Color filters.
   - *Verdict*: "Derivative / Reused Content"

VALIDATION PERSPECTIVE (DUAL CHECK):
When calculating 'transformation_score', you must validate against TWO perspectives:
1. **AS YOUTUBE AI (Bot)**: "Does this video have distinct audio/visual fingerprints from the original? Does the voiceover confuse the Content ID match?"
2. **AS HUMAN REVIEWER (Policy)**: "Did the creator add *meaning*? Is this just a repost, or is it a new creative work with critical/educational value?"

The score must satisfy BOTH. A high score means the Bot can't match it AND the Human sees value.

APPLY THESE TIERS STRICTLY. A score >90 GUARANTEES compliance with the "Critical Commentary" clause.

You MUST return valid JSON ONLY.

Schema:
{{
  "caption_final": "<DISPLAY TEXT>",
  "approved": true,
  "risk_level": "LOW|MEDIUM|HIGH",
  "risk_reason": "<SHORT 1-SENTENCE REASON>",
  "improvement_tips": ["<TIP 1>", "<TIP 2>"],
  "policy_citation": "<EXACT POLICY PHRASE MATCHED (e.g. 'Critical Commentary')>",
  "transformation_score": <0-100>,
  "verdict": "<Transformative|Derivative|High Risk>"
}}

Rules:
- caption_final MUST be the exact text to show on screen
- approved MUST be true or false
- risk_level MUST be one of LOW, MEDIUM, HIGH based on YPP safety (sexual/violent/controversial = HIGH)
- risk_reason MUST explicitly explain WHY the risk level was chosen (e.g. "Safe fashion content", "Potential skin exposure", "Explicit language")
- transformation_score represents how much the content differs from raw source (0=Raw, 100=Original). 
- verdict should be "Transformative" if score > 30, else "Derivative".
- improvement_tips MUST be an array of 2 actionable strings on how to increase the score (e.g. "Add voiceover", "More visual cuts").
- NEVER return analysis text
- NEVER return explanations
- NEVER return labels instead of captions

INPUT:
Visual Description: {input_description}
Niche: {content_origin}
Transformations Applied: {transformations}
"""

class MonetizationStrategist:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.provider = "none"
        self.model = None
        
        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
                self.model = genai.GenerativeModel(model_name)
                self.provider = "gemini"
                logger.info(f"🧠 YPP Editor Brain: ACTIVE (Model: {model_name})")
            except Exception as e:
                logger.error(f"❌ Gemini Brain Init Failed: {e}")
        else:
            logger.warning("🧠 YPP Editor Brain: INACTIVE (No Gemini Key)")

    def analyze_content(self, title: str, duration: float, transformations: Dict = {}) -> Dict:
        """
        Analyzes content using Gemini as the sole authority.
        """
        if self.provider != "gemini" or not self.model:
            return self._fallback_response(title)

        try:
            # 1. Input Sanitization (Safety First)
            # Remove control chars, strip whitespace, truncate to 200 chars
            clean_title = re.sub(r'[\x00-\x1F\x7F]', '', title).strip()
            clean_title = clean_title[:200]
            
            # Prepare Prompt
            # ORIGIN LOGIC: Standardize to public_social_media for internal logic, but concise for prompt
            origin = "public_social_media" 
            
            # Format transformation string
            trans_str = "None"
            if transformations:
                trans_str = ", ".join([f"{k}: {v}" for k,v in transformations.items()])
                
            final_prompt = EDITOR_PROMPT.format(
                input_description=clean_title, 
                content_origin=origin,
                transformations=trans_str
            )
            
            # Call Gemini
            response = self.model.generate_content(
                final_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3, 
                    response_mime_type="application/json"
                )
            )
            
            response_text = response.text.strip()
            logger.info(f"🧠 RAW GEMINI RESPONSE: {response_text}")
            return self._parse_json_response(response_text, clean_title)

        except Exception as e:
            logger.error(f"🧠 Brain Analysis Error: {e}")
            return self._fallback_response(title, error=e, transformations=transformations)

    def _parse_json_response(self, text: str, original_title: str) -> Dict:
        """
        Parses strictly JSON response with Regex extraction and Logic Validation.
        """
        try:
            # 1. Extract JSON Object (Strict Regex)
            # Look for non-greedy match between first { and last }
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if not match:
                 logger.warning("🧠 Invalid JSON format: No brackets found.")
                 return self._fallback_response(original_title, error=ValueError("Invalid JSON"))
                 
            json_str = match.group(1)
            data = json.loads(json_str)
            
            if not data.get("approved"):
                 return {
                    "final_caption": original_title, # Fallback but rejected
                    "risk_level": "HIGH",
                    "risk_reason": data.get("risk_reason", "Brain rejected content"),
                    "source": "public_social_media" 
                 }
            
            # 2. Strict Caption Validation
            caption = data.get("caption_final", "").strip()
            
            # Rule A: Length Upgrade (8-15 preferred, allow up to 25 words due to new micro-commentary)
            word_count = len(caption.split())
            if word_count < 4 or word_count > 25:
                 logger.warning(f"🧠 Validation Fail: Length ({word_count}) - '{caption}'")
                 return self._fallback_response(original_title, error=ValueError("Validation: Length"))
                 
            # Rule B: Banned Prefixes
            lower_cap = caption.lower()
            if lower_cap.startswith(("caption:", "title:", "description:", "output:")):
                 logger.warning(f"🧠 Validation Fail: Prefix - '{caption}'")
                 return self._fallback_response(original_title, error=ValueError("Validation: Prefix"))

            # Rule C: Metadata / Classifications
            banned_keywords = ["editorial", "approved", "safe context", "public domain", "ypp safe"]
            if lower_cap in banned_keywords:
                 logger.warning(f"🧠 Validation Fail: Classification Word - '{caption}'")
                 return self._fallback_response(original_title, error=ValueError("Validation: Classification"))

            # Rule D: Hashtags / Allowlist characters
            if "#" in caption or "@" in caption:
                 logger.warning(f"🧠 Validation Fail: Hashtag/Mention - '{caption}'")
                 return self._fallback_response(original_title, error=ValueError("Validation: Hashtag"))

            # Success
            return {
                "approved": True,
                "final_caption": caption, 
                "caption_style": "EDITORIAL",
                "risk_level": data.get("risk_level", "LOW"),
                "risk_reason": data.get("risk_reason", "Approved by safety filter"),
                "improvement_tips": data.get("improvement_tips", []),
                "policy_citation": data.get("policy_citation", "Analysis pending"),
                "transformation_score": data.get("transformation_score", 10),
                "verdict": data.get("verdict", "Derivative"),
                "source": "public_social_media"
            }
            
        except json.JSONDecodeError:
            logger.error(f"🧠 JSON Decode Failed: {text[:50]}...")
            return self._fallback_response(original_title, error=ValueError("JSON Decode"))
        except Exception as e:
            logger.error(f"🧠 Parsing Error: {e}")
            return self._fallback_response(original_title, error=e)

    def _fallback_response(self, caption: str, error: Exception = None, transformations: Dict = {}) -> Dict:
        """
        Returns a FAIL-SAFE response using Local Templates and Calculation.
        Calculates a "Pipeline Certified" score if Gemini is offline.
        """
        safe_cap = self.get_safe_fallback()
        
        # Default Values
        risk = "LOW"
        reason = "Brain Offline - Used Safe Template"
        
        # Check for Quota Error
        if error:
            err_str = str(error).lower()
            if "429" in err_str or "quota" in err_str:
                risk = "UNKNOWN"
                reason = "Quota Exceeded (429 Error), so Brain is Offline."
        
        # --- LOCAL PIPELINE CERTIFICATION (FALLBACK SCORING) ---
        # Replicate Tier Logic Locally
        score = 10
        verdict = "Derivative"
        tips = []

        # Tier 1: Voice + Inpaint
        has_voice = "Voiceover" in transformations
        has_inpaint = "Inpainting" in transformations
        
        if has_voice and has_inpaint:
            score = 90
            verdict = "[Tier 1] Transformative (Pipeline Certified)"
            reason += " (Score derived from verified Pipeline features)"
        # Tier 2: One of them
        elif has_voice or has_inpaint:
            score = 70
            verdict = "[Tier 2] Transformative (Weak - Pipeline Certified)"
            if not has_voice: tips.append("Add voiceover for Tier 1 score.")
            if not has_inpaint: tips.append("Remove watermark for Tier 1 score.")
        # Tier 3: Visuals only
        else:
             score = 30
             verdict = "[Tier 3] Derivative (Pipeline Certified)"
             tips.append("Add voiceover and clean watermark.")

        return {
            "approved": True, # Fallback is ostensibly safe
            "final_caption": safe_cap,
            "caption": safe_cap, # Legacy key
            "caption_style": "FALLBACK",
            "risk_level": risk, 
            "risk_reason": reason,
            "transformation_score": score,
            "verdict": verdict,
            "improvement_tips": tips,
            "policy_citation": "Fallback Logic (Gemini Unavailable)",
            "source": "public_social_media"
        }

    def generate_editorial_title(self, context: str) -> tuple:
        """
        Generates a clickbait, high-performing title AND description for compilations.
        Returns: (title, description)
        """
        fallback_title = f"Compilation: {context}"
        fallback_desc = f"Compilation of best moments for {context}. #SafeForWork #Fashion"
        
        if self.provider != "gemini" or not self.model:
             return fallback_title, fallback_desc
             
        try:
             prompt = f"""
             YOU ARE A YOUTUBE EXPERT.
             Generate ONE high-performing, clickbait TITLE and a short SEO DESCRIPTION for a compilation about: "{context}".
             
             RULES:
             1. TITLE: 5-10 words, Clickbait, Emojis ok (max 1), "Ultimate", "Best 2026", etc.
             2. DESCRIPTION: 2-3 sentences. SEO friendly. Professional. NO hashtags (we add them later).
             3. OUTPUT TYPE: Valid JSON ONLY.
             
             Schema:
             {{
                "title": "Your Title Here",
                "description": "Your description here."
             }}
             """
             
             response = self.model.generate_content(prompt)
             text = response.text.strip()
             
             # JSON Extraction
             match = re.search(r'(\{.*\})', text, re.DOTALL)
             if match:
                 data = json.loads(match.group(1))
                 title = data.get("title", fallback_title).replace('"', '').replace('*', '')
                 desc = data.get("description", fallback_desc)
                 
                 logger.info(f"🧠 Generated Title: {title}")
                 return title, desc
             else:
                 # Fallback if specific text gen (legacy support attempt) or fail
                 title = text.replace('"', '').replace('*', '')
                 return title, fallback_desc
                 
        except Exception as e:
             logger.error(f"🧠 Title/Desc Gen Failed: {e}")
             return fallback_title, fallback_desc

    def get_safe_fallback(self) -> str:
        """
        Returns a guaranteed safe caption from:
        1. Local Storage (caption_prompt.json)
        2. Hardcoded Revenue-Safe Templates
        """
        try:
            if os.path.exists("caption_prompt.json"):
                with open("caption_prompt.json", "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "caption_final" in data and len(data["caption_final"]) > 5:
                         val = data["caption_final"]
                         # Quick re-validate stored caption
                         if "#" not in val and len(val.split()) >= 2:
                             logger.info(f"🛡️ Using Stored Fallback: {val}")
                             return val
        except Exception: pass
            
        return "A quiet moment captured today"

    def save_successful_caption(self, caption: str, source: str, style: str):
        """
        Persists the safe caption to disk ATOMICALLY.
        """
        try:
            data = {
                "caption_final": caption,
                "last_source": source,
                "timestamp": datetime.now().isoformat()
            }
            
            # Atomic Write via Temp
            with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=".", encoding='utf-8') as tmp:
                json.dump(data, tmp, indent=2)
                tmp_path = tmp.name
                
            shutil.move(tmp_path, "caption_prompt.json")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save caption persistence: {e}")

# Singleton
brain = MonetizationStrategist()
