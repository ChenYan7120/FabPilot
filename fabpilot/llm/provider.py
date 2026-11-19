"""Unified LLM provider.

A single entry point (`build_llm`) returns an `LLMClient` based on
``settings.llm_provider``. Real mode talks to any OpenAI-compatible endpoint
(DeepSeek, Qwen/DashScope, etc.); offline mode uses a deterministic FakeLLM
that must never be presented as autonomous reasoning.
"""
from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

import httpx

from fabpilot.config import Settings, get_settings
from fabpilot.llm.schemas import AssistantResponse, ChatMessage, ToolCall
from fabpilot.utils import ConfigurationError, get_logger

logger = get_logger(__name__)

# A JSON Schema describing the function-calling tools available to the model.
ToolSchema = dict[str, Any]


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, messages: list[ChatMessage], tools: list[ToolSchema] | None = None) -> AssistantResponse:
        ...


class OpenAICompatibleLLM:
    """Thin client for OpenAI-compatible ``/chat/completions`` endpoints."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        if not api_key:
            raise ConfigurationError(
                "LLM API key is empty. Set FABPILOT_LLM_API_KEY, or use "
                "FABPILOT_LLM_PROVIDER=fake for offline mode."
            )

    def complete(self, messages: list[ChatMessage], tools: list[ToolSchema] | None = None) -> AssistantResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            resp = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # network / HTTP errors surface to the agent loop
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        data = resp.json()
        return _parse_chat_completion(data)


def _parse_chat_completion(data: dict[str, Any]) -> AssistantResponse:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected LLM response shape: {data}") from exc

    tool_calls: list[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {"_raw": fn.get("arguments")}
        tool_calls.append(ToolCall(id=tc.get("id", ""), name=fn.get("name", ""), arguments=arguments))

    return AssistantResponse(
        content=message.get("content") or "",
        tool_calls=tool_calls,
        finish_reason=data["choices"][0].get("finish_reason", "stop"),
    )


def build_llm(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        from fabpilot.llm.fake import FakeLLM

        return FakeLLM()
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleLLM(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
    raise ConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")
