from ..config import Settings, get_settings
from .base import DraftContext, DraftResult, SalesLLM
from .mock import MockLLM


def build_llm(settings: Settings | None = None) -> SalesLLM:
    settings = settings or get_settings()
    if settings.resolved_llm_mode == "claude":
        from .claude import ClaudeLLM

        return ClaudeLLM(settings)
    return MockLLM()


__all__ = ["DraftContext", "DraftResult", "MockLLM", "SalesLLM", "build_llm"]
