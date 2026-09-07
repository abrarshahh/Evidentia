import re
import json
import time
import asyncio
import logging
from typing import Optional, Dict, Any, List
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Dedicated Multi-Key Gemini LLM Client using official Google GenAI SDK (google-genai)
    with Round-Robin load balancing across multiple account keys, automatic per-key rate-limit
    cooldown failover, and model fallback.
    """

    def __init__(self):
        self.api_keys: List[str] = settings.get_gemini_api_keys()
        self.clients: List[genai.Client] = []
        self._cooldowns: Dict[int, float] = {}  # client_index -> unix timestamp when cooldown expires
        self._current_index = 0
        self._lock = asyncio.Lock()

        for key in self.api_keys:
            try:
                self.clients.append(genai.Client(api_key=key))
            except Exception as e:
                logger.error(f"Failed to initialize GenAI client for key {key[:8]}...: {e}")

        logger.info(f"Initialized Gemini LLMClient Pool with {len(self.clients)} active API keys.")

    def is_available(self) -> bool:
        return len(self.clients) > 0

    def get_pool_size(self) -> int:
        return len(self.clients)

    async def _get_next_client(self) -> tuple[genai.Client, int]:
        """
        Get next active client index using round-robin, avoiding clients currently in cooldown.
        If all clients are in cooldown, wait for the shortest cooldown to expire.
        """
        if not self.clients:
            raise RuntimeError("GEMINI_API_KEY is not configured in .env")

        async with self._lock:
            now = time.time()
            num_clients = len(self.clients)

            # Try to find a non-cooldowned client starting from _current_index
            for i in range(num_clients):
                idx = (self._current_index + i) % num_clients
                exp_time = self._cooldowns.get(idx, 0.0)
                if now >= exp_time:
                    self._current_index = (idx + 1) % num_clients
                    return self.clients[idx], idx

            # If all clients are in cooldown, find the one with the earliest expiration
            min_idx = min(self._cooldowns.keys(), key=lambda k: self._cooldowns[k]) if self._cooldowns else 0
            wait_seconds = max(0.5, self._cooldowns.get(min_idx, now + 1.0) - now)
            logger.warning(f"All {num_clients} Gemini API keys are rate-limited. Waiting {wait_seconds:.1f}s for key #{min_idx+1} cooldown...")

        # Sleep outside lock to avoid blocking other tasks
        await asyncio.sleep(wait_seconds)

        async with self._lock:
            self._current_index = (min_idx + 1) % len(self.clients)
            return self.clients[min_idx], min_idx

    def _mark_cooldown(self, client_idx: int, cooldown_seconds: float = 15.0):
        """Mark a specific client index on cooldown for rate limits."""
        self._cooldowns[client_idx] = time.time() + cooldown_seconds
        logger.warning(
            f"Gemini API key #{client_idx+1}/{len(self.clients)} hit 429 rate limit. Placed on {cooldown_seconds}s cooldown."
        )

    async def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
    ) -> str:
        """
        Generate text using official Google GenAI SDK with multi-key round-robin rotation,
        instant 429 key failover, and model fallback.
        """
        if not self.clients:
            raise RuntimeError("No valid GEMINI_API_KEY configured in .env")

        config = None
        if system_instruction:
            config = types.GenerateContentConfig(system_instruction=system_instruction)

        active_models = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-1.5-flash", "gemini-2.0-flash"]
        models_to_try = [model_name] + [m for m in active_models if m != model_name]

        max_attempts_per_model = max(3, len(self.clients) * 2)

        for m in models_to_try:
            for attempt in range(max_attempts_per_model):
                client, client_idx = await self._get_next_client()
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=prompt,
                        config=config,
                    )
                    if response and response.text:
                        return response.text.strip()
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        self._mark_cooldown(client_idx, cooldown_seconds=15.0)
                        # Instant failover to next key in pool on next iteration
                        continue
                    else:
                        logger.warning(f"Gemini model {m} failed on key #{client_idx+1}: {e}. Trying fallback model...")
                        break

        raise RuntimeError("All Gemini API keys and model fallbacks failed. Check GEMINI_API_KEYS or quota limits.")

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
