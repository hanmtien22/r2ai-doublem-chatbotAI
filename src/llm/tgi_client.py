import requests
import json
import logging
import time
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GenericLLMClient:
    """Client hỗ trợ gọi API chuẩn OpenAI (vLLM, Ollama, API Cloud).

    Hai method chính:
    - generate(prompt)         : single user message → trả về str
    - generate_chat(system, user) : system + user message → trả về str
    """

    def __init__(
        self,
        endpoint: str = "https://openrouter.ai/api/v1",
        api_key: str = "EMPTY",
        model: str = "qwen/qwen-2.5-7b-instruct:free",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 120,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.chat_url = f"{self.endpoint}/chat/completions"
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    #  Internal helper                                                     #
    # ------------------------------------------------------------------ #
    def _post_chat(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """Gọi chat/completions và trả về content string. Raise nếu thất bại."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.chat_url, headers=headers, json=payload, timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code in (429, 503):
                    last_error = e
                    wait = self.retry_delay * (2**attempt)
                    logger.warning(
                        "LLM server overloaded (attempt %d/%d), retry in %.1fs…",
                        attempt + 1, self.max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                logger.error("LLM HTTP error: %s", e)
                raise

            except Exception as e:
                last_error = e
                logger.error(
                    "LLM Client error (attempt %d/%d): %s",
                    attempt + 1, self.max_retries, e,
                )
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(
            f"LLM failed after {self.max_retries} retries: {last_error}"
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_mode: bool = False,
        schema: any = None,
        **kwargs,
    ) -> str:
        if schema is not None:
            json_mode = True
        """Single user-message call → trả về str."""
        messages = [{"role": "user", "content": prompt}]
        try:
            return self._post_chat(messages, max_tokens, temperature, json_mode)
        except Exception as e:
            logger.error("generate() failed: %s", e)
            return ""

    def generate_chat(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        """System + User message call → trả về str."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            return self._post_chat(messages, max_tokens, temperature, json_mode)
        except Exception as e:
            logger.error("generate_chat() failed: %s", e)
            return ""


class OllamaClient(GenericLLMClient):
    """Preset cho Ollama local (RTX 3050 6 GB VRAM).

    Model gợi ý theo VRAM:
    - qwen2.5:3b   → ~2.0 GB  (nhanh, phù hợp cho routing & answer)
    - qwen2.5:7b   → ~4.5 GB  (chất lượng cao hơn, vừa đủ VRAM)
    - phi3.5:mini  → ~2.4 GB  (tiếng Anh tốt)
    - llama3.2:3b  → ~2.0 GB

    Mặc định dùng qwen2.5:3b — an toàn nhất cho 6 GB.
    """

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        host: str = "http://localhost:11434",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 180,
    ):
        super().__init__(
            endpoint=f"{host}/v1",
            api_key="ollama",          # Ollama không cần key thật
            model=model,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
        )
        logger.info("OllamaClient initialized: model=%s  endpoint=%s", model, self.endpoint)


class TGIClient(GenericLLMClient):
    """Alias backward-compat."""
    pass