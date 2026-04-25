from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp
from openai import OpenAI

# ----------------------------
# 1) Provider-agnostic LLM API
# ----------------------------

@dataclass
class LLMResponse:
    teammate_type: str
    confidence: float
    rationale: str
    raw_text: str


class LLMClient(Protocol):
    """Minimal interface so CoLLAB logic doesn't care about provider."""
    def classify_teammate_type(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        ...


class OpenAIChatClient:
    """
    OpenAI implementation (Responses API or Chat Completions API depending on your SDK).
    This is intentionally lightweight; adapt to your preferred OpenAI SDK style.

    Env var:
      OPENAI_API_KEY
    """
    def __init__(self, model: str):
        self.model = model


class ChatClient:
    """
    HuggingFace implementation (local transformers).
    Works with Qwen/LLaMA/etc. IF you can load them locally.

    You can also replace this with a vLLM client, TGI, etc.
    """
    def __init__(self, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
        # Check for OpenRouter first (via OPENROUTER_API_KEY)
        if os.environ.get("OPENROUTER_API_KEY"):
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            if model_name == "deepseek-ai/DeepSeek-V3.2:preferred":
                print("Using OpenRouter with DeepSeek-V3.2:preferred, which maps to deepseek/deepseek-v3.2")
                model_name = "deepseek/deepseek-v3.2"
            self.model_name = model_name
            self.is_openai = False
            self.provider = "openrouter"
        elif model_name.startswith("openai"):
            self.client = OpenAI()
            self.model_name = model_name.split("/")[-1]
            self.is_openai = True
            self.provider = "openai"
        else:
            # Fallback to HuggingFace
            self.client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=os.environ["HF_API_TOKEN"],
            )
            self.model_name = model_name
            self.is_openai = False
            self.provider = "huggingface"

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ]
        
        if self.is_openai:
            if self.model_name == "gpt-5-nano":
                temperature = 0.01

            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "max_completion_tokens": max_tokens,
            }
            
            # Add JSON schema if provided
            if json_schema is not None:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            
            completion = self.client.chat.completions.create(**kwargs)
        else:
            kwargs = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            
            # OpenRouter-specific optimizations
            if self.provider == "openrouter":
                # Add extra_body for OpenRouter-specific features
                kwargs["extra_body"] = {
                    # Prioritize speed over cost
                    "provider": {
                        "order": ["DeepInfra", "Together", "Lepton"],  # Fast providers
                        "allow_fallbacks": True
                    }
                }
            
            # Add JSON schema if provided
            if json_schema is not None:
                # print("Using JSON schema with HuggingFace client: ", json_schema)
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema
                }
            
            completion = self.client.chat.completions.create(**kwargs)

        return completion.choices[0].message.content


def _parse_llm_json(text: str) -> LLMResponse:
    """
    Robust-ish JSON extraction. If your models support strict JSON mode, use that.
    """
    print("Response: ", text)
    raw = text.strip()

    # Try direct JSON parse
    try:
        obj = json.loads(raw)
    except Exception:
        # Try to find the first {...} block
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"LLM did not return JSON. Raw:\n{raw}")
        obj = json.loads(raw[start : end + 1])

    teammate_type = str(obj.get("teammate_type", "")).strip()
    confidence = float(obj.get("confidence", 0.0))
    rationale = str(obj.get("rationale", "")).strip()

    return LLMResponse(
        teammate_type=teammate_type,
        confidence=confidence,
        rationale=rationale,
        raw_text=raw,
    )