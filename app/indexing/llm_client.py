import re
import json
import logging
from typing import Optional, Dict, Any
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Dedicated Gemini LLM Client using official Google GenAI SDK (google-genai)
    targeting active production models (gemini-3.6-flash, gemini-3.5-flash, gemini-2.5-pro).
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
        Generate text using official Google GenAI SDK with fallback across Gemini models.
        """
        if not self.genai_client:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        # Try primary model: gemini-3.6-flash
        try:
            response = self.genai_client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini model {model_name} failed: {e}. Trying gemini-3.5-flash...")

        # Fallback 1: gemini-3.5-flash
        try:
            response = self.genai_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as ex:
            logger.warning(f"Gemini fallback gemini-3.5-flash failed: {ex}. Trying gemini-2.5-pro...")

        # Fallback 2: gemini-2.5-pro
        try:
            response = self.genai_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=config,
            )
            if response and response.text:
                return response.text.strip()
        except Exception as ex:
            logger.error(f"Gemini fallback gemini-2.5-pro failed: {ex}")

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
