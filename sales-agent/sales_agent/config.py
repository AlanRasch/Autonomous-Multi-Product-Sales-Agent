"""Runtime configuration, loaded from environment variables or a .env file."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="SALES_AGENT_", extra="ignore")

    # "auto" uses Claude when Anthropic credentials are present, else the offline mock LLM.
    llm_mode: Literal["auto", "claude", "mock"] = "auto"
    model: str = "claude-opus-5-5"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    max_tokens: int = 16000
    # Server-side refusal fallback (Claude API only). Disable on Bedrock/Vertex/Foundry.
    enable_fallbacks: bool = True
    max_tool_iterations: int = 8

    # Persistence for the local CRM store; empty = in-memory only.
    crm_store_path: str = ""
    booking_base_url: str = "https://cal.example.com/book"

    # Optional live market search. Without a key, market intel comes from mock data.
    tavily_api_key: str = ""

    @property
    def resolved_llm_mode(self) -> Literal["claude", "mock"]:
        if self.llm_mode != "auto":
            return self.llm_mode
        has_creds = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
        return "claude" if has_creds else "mock"


@lru_cache
def get_settings() -> Settings:
    return Settings()
