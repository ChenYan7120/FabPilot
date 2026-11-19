"""集中式配置。

所有设置都通过 ``FABPILOT_`` 前缀从环境变量 / `.env` 文件中读取，因此应用行为可以
集中在一处控制。任何密钥都不应提交到代码仓库。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FABPILOT_",
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    llm_provider: str = "openai_compatible"  # openai_compatible | fake
    llm_base_url: str = "https://api.deepseek.com/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 2048

    # ---- Embedding ----
    embedding_provider: str = "hash"  # openai_compatible | hash
    embedding_base_url: str = "https://api.deepseek.com/v1"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1024

    # ---- 业务数据库 ----
    database_url: str = "postgresql+psycopg://fabpilot:fabpilot@localhost:5432/fabpilot"

    # ---- 向量存储 ----
    vector_store: str = "memory"  # memory | pgvector
    pgvector_connection: str = "postgresql+psycopg://fabpilot:fabpilot@localhost:5432/fabpilot"

    # ---- ReAct 预算 / 循环安全 ----
    supervisor_max_iterations: int = 10
    agent_max_iterations: int = 8
    tool_budget: int = 30
    loop_detection_window: int = 4

    # ---- 预测 ----
    default_forecast_model: str = "statistical_baseline"
    forecast_lookback: int = 96
    forecast_horizon: int = 16

    # ---- 路径 ----
    knowledge_base_dir: str = "knowledge_base"
    data_dir: str = "data"
    log_level: str = "INFO"
    checkpoint_sqlite: str = ""  # 可选的 sqlite 路径，用于 LangGraph checkpoint（"" = 内存存储）

    @property
    def knowledge_base_path(self) -> Path:
        p = Path(self.knowledge_base_dir)
        return p if p.is_absolute() else ROOT_DIR / p

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        return p if p.is_absolute() else ROOT_DIR / p


@lru_cache
def get_settings() -> Settings:
    return Settings()
