"""High-level entry point: runs one inbound event through the state machine."""

from __future__ import annotations

import uuid

from .config import Settings, get_settings
from .graph import build_graph
from .llm import SalesLLM, build_llm
from .models import ChatRequest, ChatResponse


class SalesAgent:
    def __init__(self, llm: SalesLLM | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_llm(self.settings)
        self.graph = build_graph(self.llm)

    async def handle(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or uuid.uuid4().hex
        state = await self.graph.ainvoke(
            {"request": request, "session_id": session_id, "tool_calls": [], "evidence": [], "trace": []}
        )
        return ChatResponse(
            session_id=session_id,
            customer_id=state["customer_id"],
            reply=state["reply"],
            intent=state["ingestion"],
            analysis=state["analysis"],
            strategy=state["strategy"],
            quote=state.get("quote"),
            meeting=state.get("meeting"),
            crm=state.get("crm"),
            tool_calls=state["tool_calls"],
            guardrail_flags=state.get("guardrail_flags", []),
            trace=state["trace"],
            llm_mode=self.llm.mode,
        )
