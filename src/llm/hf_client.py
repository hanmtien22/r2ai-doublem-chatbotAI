"""LLM chạy thẳng trong tiến trình bằng transformers — dùng cho Kaggle.

Kaggle không có sẵn Ollama và cũng không tiện dựng server trong notebook, nên
backend này nạp model vào cùng tiến trình và giữ nguyên interface
`generate` / `generate_chat` của `GenericLLMClient`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Model mặc định: bản Instruct của chính model đang dùng ở local (qwen2.5:3b)
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"


def _load_model(auto_class, model_name: str, torch_dtype, device: str, trust_remote_code: bool):
    """Nạp model, chịu được khác biệt API giữa các phiên bản transformers.

    transformers 5 đổi `torch_dtype` thành `dtype`, và `device_map` cần gói
    `accelerate`. Thử lần lượt từ bộ tham số đầy đủ tới tối giản thay vì ghim
    một phiên bản — Kaggle thường dùng 4.x, máy local có thể là 5.x.
    """
    attempts = [
        {"dtype": torch_dtype, "device_map": device if device == "cuda" else None},
        {"torch_dtype": torch_dtype, "device_map": device if device == "cuda" else None},
        {"dtype": torch_dtype},
        {"torch_dtype": torch_dtype},
        {},
    ]
    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return auto_class.from_pretrained(
                model_name, trust_remote_code=trust_remote_code, **kwargs
            )
        except Exception as e:
            # TypeError: phiên bản này không nhận tham số đó
            # ImportError/ValueError: thiếu `accelerate` cho `device_map`
            last_error = e
            logger.debug("Nạp model với %s không thành công: %s", list(kwargs), e)
    raise RuntimeError(f"Không nạp được model {model_name}: {last_error}")


class HFLocalClient:
    """Sinh text bằng transformers, không cần HTTP server."""

    def __init__(
        self,
        model: str = DEFAULT_HF_MODEL,
        device: Optional[str] = None,
        dtype: str = "auto",
        max_input_tokens: int = 3072,
        trust_remote_code: bool = False,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.max_input_tokens = max_input_tokens

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        logger.info("Đang nạp model %s lên %s (%s)...", model, device, torch_dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        self._model = _load_model(
            AutoModelForCausalLM, model, torch_dtype, device, trust_remote_code
        )
        if self._model.device.type != device:
            self._model = self._model.to(device)
        self._model.eval()

        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._torch = torch
        logger.info("Đã nạp xong %s", model)

    def _chat(self, messages: list[dict], max_tokens: int, temperature: float) -> str:
        tokenizer, model, torch = self._tokenizer, self._model, self._torch

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(model.device)

        # temperature=0 -> greedy; transformers cảnh báo nếu truyền temperature khi do_sample=False
        do_sample = temperature > 0
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature

        with torch.no_grad():
            output = model.generate(**inputs, **generate_kwargs)

        # Chỉ lấy phần model sinh thêm, bỏ lại prompt
        generated = output[0][inputs["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        try:
            return self._chat([{"role": "user", "content": prompt}], max_tokens, temperature)
        except Exception as e:
            logger.error("HFLocalClient.generate() lỗi: %s", e)
            return ""

    def generate_chat(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int = 512,
        temperature: float = 0.1,
        json_mode: bool = False,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        try:
            return self._chat(messages, max_tokens, temperature)
        except Exception as e:
            logger.error("HFLocalClient.generate_chat() lỗi: %s", e)
            return ""
