"""LLM 抽象：消息 schema、OpenAI 兼容 provider 以及确定性的 FakeLLM。"""
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
