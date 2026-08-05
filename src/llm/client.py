from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model_name: str = "Qwen2.5-14B-Instruct",
        model_path: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.0,
        timeout: int = 30,
        max_retries: int = 2,
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
            from vllm import LLM, SamplingParams  # noqa: F401
            self._engine = LLM(model=self.model_path or self.model_name)
            self._backend = "vllm"
            logger.info("Loaded vLLM backend")
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
                if self._backend == "vllm":
                    return self._generate_vllm(prompt, _max_tokens, _temperature)
                elif self._backend == "llama_cpp":
                    return self._generate_llama_cpp(prompt, _max_tokens, _temperature)
                else:
                    return self._generate_mock(prompt)
            except Exception as e:
                logger.warning("LLM generate attempt %d failed: %s", attempt + 1, e)
                if attempt == self.max_retries:
                    raise
        return ""

    def _generate_vllm(self, prompt: str, max_tokens: int, temperature: float) -> str:
        from vllm import SamplingParams
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = self._engine.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

    def _generate_llama_cpp(self, prompt: str, max_tokens: int, temperature: float) -> str:
        output = self._engine(prompt, max_tokens=max_tokens, temperature=temperature)
        return output["choices"][0]["text"].strip()

    def _generate_mock(self, prompt: str) -> str:
        logger.info("Mock LLM called with prompt length=%d", len(prompt))
        return "single_lookup"

    def generate_json(self, prompt: str, max_tokens: Optional[int] = None) -> dict[str, Any]:
        raw = self.generate(prompt, max_tokens=max_tokens)
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM JSON output: %s", raw[:200])
            return {}

    def generate_batch(self, prompts: list[str], max_tokens: Optional[int] = None) -> list[str]:
        self._load_engine()
        _max_tokens = max_tokens or self.max_tokens

        if self._backend == "vllm":
            from vllm import SamplingParams
            params = SamplingParams(max_tokens=_max_tokens, temperature=self.temperature)
            outputs = self._engine.generate(prompts, params)
            return [o.outputs[0].text.strip() for o in outputs]
        else:
            return [self.generate(p, max_tokens=_max_tokens) for p in prompts]
