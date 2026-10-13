"""LLM backend interface shared by the Claude and offline mock implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..models import (
    AnalysisResult,
    CalendarLink,
    EnrichmentBundle,
    IngestionResult,
    Interaction,
    Quote,
    StrategyDecision,
    ToolCallRecord,
)


@dataclass
class DraftContext:
    message: str
    customer_id: str
    history: list[Interaction]
    ingestion: IngestionResult
    enrichment: EnrichmentBundle
    analysis: AnalysisResult
    strategy: StrategyDecision
    meeting: CalendarLink | None = None
    feedback: str | None = None  # guardrail feedback for a revision pass


@dataclass
class DraftResult:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_outputs: list[str] = field(default_factory=list)  # raw tool text, used by guardrails
    quote: Quote | None = None
    meeting: CalendarLink | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class SalesLLM(Protocol):
    mode: str

    async def classify(self, message: str, history: list[Interaction]) -> IngestionResult: ...

    async def draft(self, ctx: DraftContext) -> DraftResult: ...
