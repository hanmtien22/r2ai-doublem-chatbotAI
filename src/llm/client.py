"""
LLMClient — alias mặc định trỏ tới OllamaClient.

Dùng cho RTX 3050 6 GB VRAM:
  - model mặc định : qwen2.5:3b  (~2 GB VRAM, Q4)
  - endpoint       : http://localhost:11434/v1  (Ollama local)

Để dùng model lớn hơn (nếu đủ VRAM):
  LLMClient(model="qwen2.5:7b")
"""
import logging
from src.llm.tgi_client import OllamaClient

logger = logging.getLogger(__name__)

# LLMClient là OllamaClient — toàn bộ codebase import LLMClient vẫn hoạt động
LLMClient = OllamaClient
