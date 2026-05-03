"""Global settings and configuration using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = Field(default="your-api-key-here", env="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # Agent
    max_iterations: int = 3
    critic_threshold: float = 0.7
    execution_timeout: int = 30

    # RAG
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_k_retrieval: int = 10
    top_k_reranked: int = 5
    chunk_size: int = 512
    chunk_overlap: int = 64

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list = ["*"]

    # Caching
    enable_cache: bool = True
    cache_ttl: int = 3600

    # Logging
    log_level: str = "INFO"
    log_file: str = "agent_system.log"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
