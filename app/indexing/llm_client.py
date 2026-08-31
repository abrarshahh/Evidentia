import re
import json
import asyncio
import logging
from typing import Optional, Dict, Any
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Dedicated Gemini LLM Client using official Google GenAI SDK (google-genai)
    targeting active production models (gemini-3.6-flash, gemini-3.5-flash, gemini-2.5-pro, gemini-2.5-flash-lite)
    with automatic 429 rate limit backoff.
    """

    def __init__(self):
        self.gemini_key = settings.get_gemini_api_key()
        self.genai_client = None
        if self.gemini_key:
            self.genai_client = genai.Client(api_key=self.gemini_key)

    def is_available(self) -> bool:
        return bool(self.gemini_key)

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
    ) -> str:
        """
        Generate text using official Google GenAI SDK with fallback across Gemini models and rate limit handling.
        """
        if not self.genai_client:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        models_to_try = [model_name, "gemini-3.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]

        for m in models_to_try:
            for attempt in range(2):
                try:
                    response = self.genai_client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        logger.warning(f"Rate limit on model {m} (attempt {attempt+1}): waiting 3s...")
                        await asyncio.sleep(3)
                    else:
                        logger.warning(f"Gemini model {m} failed: {e}. Trying fallback model...")
                        break

        raise RuntimeError("All Gemini model requests failed. Check GEMINI_API_KEY or model availability.")

    async def generate_json(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output from Gemini.
        """
        json_instruction = (
            (system_instruction or "") + "\nRespond strictly in valid JSON format without markdown code fences."
        ).strip()

        raw_text = await self.generate_text(
            prompt=prompt,
            system_instruction=json_instruction,
            model_name=model_name,
        )

        cleaned_text = raw_text.strip()
        if cleaned_text.startswith("```"):
            cleaned_text = re.sub(r"^```(?:json)?\n?", "", cleaned_text)
            cleaned_text = re.sub(r"\n?```$", "", cleaned_text)

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini JSON response: {e}. Raw response: {raw_text}")
            raise ValueError(f"Invalid JSON returned by Gemini: {str(e)}")


llm_client = LLMClient()
