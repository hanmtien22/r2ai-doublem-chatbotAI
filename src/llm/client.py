from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model_name: str = "qwen2.5:3b",
        model_path: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.max_retries = max_retries
        self._engine = None

    def _load_engine(self) -> None:
        if self._engine is not None:
            return
        logger.info("Loading LLM engine: %s", self.model_name)
        try:
            import ollama
            self._engine = ollama
            self._backend = "ollama"
            logger.info("Loaded ollama backend")
        except ImportError:
            try:
                from llama_cpp import Llama  # noqa: F401
                if self.model_path:
                    self._engine = Llama(model_path=self.model_path, n_ctx=4096)
                    self._backend = "llama_cpp"
                    logger.info("Loaded llama.cpp backend")
                else:
                    logger.warning("No model_path for llama.cpp, using mock")
                    self._engine = "mock"
                    self._backend = "mock"
            except ImportError:
                logger.warning("No LLM backend available, using mock")
                self._engine = "mock"
                self._backend = "mock"

    def generate(self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None) -> str:
        self._load_engine()
        _max_tokens = max_tokens or self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries + 1):
            try:
                if self._backend == "ollama":
                    return self._generate_ollama(prompt, _max_tokens, _temperature)
                elif self._backend == "llama_cpp":
                    return self._generate_llama_cpp(prompt, _max_tokens, _temperature)
                else:
                    return self._generate_mock(prompt)
            except Exception as e:
                logger.warning("LLM generate attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    raise
        return ""

    def generate_chat(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Gọi chat API (system/user roles). Fallback về generate() nếu backend không hỗ trợ."""
        self._load_engine()
        _max_tokens = max_tokens or self.max_tokens
        _temperature = temperature if temperature is not None else self.temperature

        for attempt in range(self.max_retries + 1):
            try:
                if self._backend == "ollama":
                    return self._chat_ollama(system_prompt, user_message, _max_tokens, _temperature)
                else:
                    combined = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
                    return self.generate(combined, max_tokens=_max_tokens, temperature=_temperature)
            except Exception as e:
                logger.warning("LLM chat attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    combined = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
                    return self.generate(combined, max_tokens=_max_tokens, temperature=_temperature)
        return ""

    def _chat_ollama(self, system_prompt: str, user_message: str, max_tokens: int, temperature: float) -> str:
        response = self._engine.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["message"]["content"].strip()

    def _generate_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        response = self._engine.generate(
            model=self.model_name,
            prompt=prompt,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
        return response["response"].strip()

    def _generate_llama_cpp(self, prompt: str, max_tokens: int, temperature: float) -> str:
        output = self._engine(prompt, max_tokens=max_tokens, temperature=temperature)
        return output["choices"][0]["text"].strip()

    def _generate_mock(self, prompt: str) -> str:
        logger.info("Mock LLM called with prompt length=%d", len(prompt))
        return "single_lookup"

    def generate_json(self, prompt: str, max_tokens: Optional[int] = None) -> dict[str, Any]:
        raw = self.generate(prompt, max_tokens=max_tokens)
        return self._parse_json_robust(raw)

    def generate_json_chat(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        raw = self.generate_chat(system_prompt, user_message, max_tokens=max_tokens)
        return self._parse_json_robust(raw)

    @staticmethod
    def _parse_json_robust(raw: str) -> dict[str, Any]:
        """Parse JSON từ LLM output. Strip markdown fences, fallback extract { } nếu bị bọc trong text."""
        text = raw.strip()

        if text.startswith("```"):
            inner_lines = []
            for line in text.split("\n")[1:]:
                if line.strip() == "```":
                    break
                inner_lines.append(line)
            text = "\n".join(inner_lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for pattern in (r"\{.*\}", r"\[.*\]"):
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        logger.warning("Failed to parse LLM JSON output: %s", raw[:300])
        return {}

    def generate_batch(self, prompts: list[str], max_tokens: Optional[int] = None) -> list[str]:
        self._load_engine()
        _max_tokens = max_tokens or self.max_tokens
        return [self.generate(p, max_tokens=_max_tokens) for p in prompts]
