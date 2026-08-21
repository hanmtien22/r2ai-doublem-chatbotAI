"""Chọn backend LLM theo môi trường đang chạy.

- `ollama` : máy local đã có `ollama serve` (mặc định khi chạy ở máy cá nhân)
- `hf`     : nạp model bằng transformers ngay trong tiến trình (Kaggle, Colab)
- `none`   : không dùng LLM — chỉ chạy đường tra cứu tất định
- `auto`   : dò theo thứ tự Ollama -> HF -> none
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

BACKENDS = ("auto", "ollama", "hf", "none")

DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


class NullLLMClient:
    """Không gọi LLM, trả chuỗi rỗng ngay lập tức.

    Mọi nơi gọi LLM trong pipeline đều đã xử lý trường hợp trả về rỗng, nên
    dùng client này pipeline vẫn chạy và chỉ trả lời bằng đường tra cứu tất
    định. Quan trọng là nó trả về NGAY: nếu để `OllamaClient` gọi vào một
    endpoint không tồn tại thì mỗi câu hỏi phải chờ hết 3 lần retry.
    """

    model = "none"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.1,
                 json_mode: bool = False) -> str:
        return ""

    def generate_chat(self, system_prompt: str, user_message: str, max_tokens: int = 512,
                      temperature: float = 0.1, json_mode: bool = False) -> str:
        return ""


def _ollama_is_up(host: str, timeout: float = 2.0) -> bool:
    try:
        import requests

        response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def _has_gpu() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def build_llm_client(
    backend: str = "auto",
    model: Optional[str] = None,
    host: str = DEFAULT_OLLAMA_HOST,
    **kwargs: Any,
):
    """Dựng client LLM theo backend yêu cầu.

    Đọc mặc định từ biến môi trường R2AI_LLM_BACKEND / R2AI_LLM_MODEL để
    notebook có thể đổi backend mà không phải sửa code.
    """
    backend = (backend or "auto").lower()
    if backend == "auto":
        backend = os.environ.get("R2AI_LLM_BACKEND", "auto").lower()
    if model is None:
        model = os.environ.get("R2AI_LLM_MODEL")

    if backend not in BACKENDS:
        raise ValueError(f"backend không hợp lệ: {backend!r}, chọn một trong {BACKENDS}")

    if backend == "none":
        logger.info("LLM backend: none — chỉ dùng đường tra cứu tất định")
        return NullLLMClient()

    if backend == "ollama":
        from src.llm.tgi_client import OllamaClient

        return OllamaClient(model=model or DEFAULT_OLLAMA_MODEL, host=host, **kwargs)

    if backend == "hf":
        from src.llm.hf_client import DEFAULT_HF_MODEL, HFLocalClient

        return HFLocalClient(model=model or DEFAULT_HF_MODEL, **kwargs)

    # auto: ưu tiên Ollama nếu server đang chạy, sau đó tới transformers
    if _ollama_is_up(host):
        from src.llm.tgi_client import OllamaClient

        logger.info("LLM backend: ollama (phát hiện server tại %s)", host)
        return OllamaClient(model=model or DEFAULT_OLLAMA_MODEL, host=host, **kwargs)

    try:
        from src.llm.hf_client import DEFAULT_HF_MODEL, HFLocalClient

        logger.info("LLM backend: hf (transformers, gpu=%s)", _has_gpu())
        return HFLocalClient(model=model or DEFAULT_HF_MODEL, **kwargs)
    except Exception as e:
        logger.warning("Không dựng được backend transformers (%s) — chuyển sang chế độ không LLM", e)
        return NullLLMClient()
