"""LLM client abstraction (OpenAI Chat Completions, JSON mode, text or image input)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass
class LLMResponse:
    data: dict
    raw: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    model: str

    def complete_json(self, messages: list[dict]) -> LLMResponse: ...


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class OpenAIClient:
    def __init__(self, model: str, api_key: str | None = None, timeout_s: float = 90.0) -> None:
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(api_key=api_key, timeout=timeout_s)

    def complete_json(self, messages: list[dict]) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, response_format={"type": "json_object"}
        )
        content = resp.choices[0].message.content or "{}"
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            data=parse_json_object(content),
            raw=content,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
