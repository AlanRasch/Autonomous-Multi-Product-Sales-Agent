"""Shared state passed between the five graph nodes."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from ..models import (
    AnalysisResult,
    CalendarLink,
    ChatRequest,
    CrmRecord,
    EnrichmentBundle,
    IngestionResult,
    Interaction,
    Quote,
    StageTrace,
    StrategyDecision,
    ToolCallRecord,
)


class AgentState(TypedDict, total=False):
    request: ChatRequest
    session_id: str
    customer_id: str
    history: list[Interaction]

    ingestion: IngestionResult  # STATE 1
    enrichment: EnrichmentBundle  # STATE 2
    analysis: AnalysisResult  # STATE 3
    strategy: StrategyDecision  # STATE 4
    reply: str  # STATE 5
    quote: Quote | None
    meeting: CalendarLink | None
    crm: CrmRecord | None
    guardrail_flags: list[str]

    # Append-only audit trails (LangGraph reducers concatenate node outputs).
    tool_calls: Annotated[list[ToolCallRecord], operator.add]
    evidence: Annotated[list[str], operator.add]  # raw tool output text, for guardrails
    trace: Annotated[list[StageTrace], operator.add]
