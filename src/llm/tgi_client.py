import requests
import json
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class GenericLLMClient:
    """Client hỗ trợ gọi API chuẩn OpenAI (vLLM, Ollama, API Cloud) hoặc HuggingFace."""
    def __init__(self, endpoint: str = "https://openrouter.ai/api/v1", api_key: str = "EMPTY", model: str = "qwen/qwen-2.5-7b-instruct:free"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.chat_url = f"{self.endpoint}/chat/completions"

    def generate(self, prompt: str, schema: Optional[BaseModel] = None, max_new_tokens: int = 2048, temperature: float = 0.1) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_new_tokens,
            "temperature": temperature,
        }
        
        # vLLM/Ollama/Groq/OpenAI hỗ trợ response_format = json_object
        if schema:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = requests.post(self.chat_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            generated_text = result["choices"][0]["message"]["content"]
            
            if schema:
                try:
                    return json.loads(generated_text)
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse LLM response as JSON. Response: {generated_text}")
                    return {"error": "Invalid JSON response", "raw": generated_text}
            
            return {"text": generated_text}
            
        except Exception as e:
            logger.error(f"LLM Client error: {str(e)}")
            return {"error": str(e)}

class TGIClient(GenericLLMClient):
    pass