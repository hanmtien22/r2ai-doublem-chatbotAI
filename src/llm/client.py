import logging
from typing import Optional

logger = logging.getLogger(__name__)

class LLMClient:
    """Mock/Wrapper for legacy local client."""
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, prompt: str, max_tokens: int = 100) -> str:
        return ""
