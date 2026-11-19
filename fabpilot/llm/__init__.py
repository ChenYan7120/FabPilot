"""LLM abstraction: message schemas, OpenAI-compatible provider and deterministic FakeLLM."""
from __future__ import annotations

from fabpilot.llm.fake import FakeLLM
from fabpilot.llm.provider import LLMClient, OpenAICompatibleLLM, build_llm
from fabpilot.llm.schemas import AssistantResponse, ChatMessage, ToolCall

__all__ = [
    "FakeLLM",
    "LLMClient",
    "OpenAICompatibleLLM",
    "build_llm",
    "AssistantResponse",
    "ChatMessage",
    "ToolCall",
]
