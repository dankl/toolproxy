"""
Upstream LLM HTTP client with retry logic.
"""
import httpx
import asyncio
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class VLLMClient:
    """HTTP client for an OpenAI-compatible upstream LLM API."""

    def __init__(self, base_url: str, model: str, api_key: str = "dummy-key", timeout: int = 180, max_retries: int = 3, retry_on_timeout: bool = False):
        """
        Initialize upstream LLM client.

        Args:
            base_url: Upstream API base URL (OpenAI-compatible)
            model: Model name
            api_key: API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_on_timeout = retry_on_timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"Authorization": f"Bearer {api_key}"}
        )

    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Call upstream chat completion API.

        Args:
            messages: List of chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice setting
            **kwargs: Additional parameters

        Returns:
            Response dict from upstream

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        url = f"{self.base_url}/chat/completions"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tools:
            payload["tools"] = tools
            logger.debug(f"Added {len(tools)} tools to upstream request")

        if tool_choice:
            payload["tool_choice"] = tool_choice
            logger.debug(f"Added tool_choice to upstream request: {tool_choice}")

        # Add any additional parameters
        payload.update(kwargs)

        # Retry logic with exponential backoff
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"upstream request attempt {attempt + 1}/{self.max_retries}")
                logger.debug(f"Payload: {payload}")

                response = await self.client.post(url, json=payload)
                response.raise_for_status()

                result = response.json()
                logger.debug(f"upstream response: {result}")
                return result

            except httpx.TimeoutException as e:
                logger.warning(f"upstream timeout on attempt {attempt + 1}: {str(e)}")
                # By default do NOT retry on timeout — OCI timeouts typically mean
                # the model is stuck, not a transient network blip. Retrying just
                # multiplies the user-visible wait time.
                if not self.retry_on_timeout or attempt == self.max_retries - 1:
                    raise
                # Exponential backoff
                await asyncio.sleep(2 ** attempt)

            except httpx.HTTPStatusError as e:
                logger.error(f"upstream HTTP error {e.response.status_code}: {e.response.text}")
                # Retry on 5xx (transient upstream errors)
                if e.response.status_code >= 500 and attempt < self.max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(f"Upstream 5xx — retrying in {wait}s (attempt {attempt + 1}/{self.max_retries})")
                    await asyncio.sleep(wait)
                    continue
                raise

            except httpx.HTTPError as e:
                logger.error(f"upstream HTTP error on attempt {attempt + 1}: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
