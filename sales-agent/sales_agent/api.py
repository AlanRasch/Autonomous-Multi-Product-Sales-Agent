"""FastAPI app exposing the sales agent.

Run:  uvicorn sales_agent.api:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import anthropic
from fastapi import FastAPI, HTTPException

from . import __version__
from .agent import SalesAgent
from .models import ChatRequest, ChatResponse, ConversationThread, CrmRecord
from .store import get_store
from .tools.customer import get_interaction_history

log = logging.getLogger("sales_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = SalesAgent()
    log.info("Sales agent ready (llm_mode=%s)", app.state.agent.llm.mode)
    yield


app = FastAPI(
    title="Autonomous Multi-Product Sales Agent",
    version=__version__,
    description="5-stage agent graph: Ingestion -> Enrichment -> Analysis -> Strategy -> Execution.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "llm_mode": app.state.agent.llm.mode, "version": __version__}


@app.post("/api/v1/agent/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await app.state.agent.handle(request)
    except anthropic.RateLimitError as exc:
        raise HTTPException(status_code=429, detail="LLM rate limited, retry shortly") from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(status_code=503, detail="LLM unreachable") from exc
    except anthropic.APIStatusError as exc:
        log.error("Anthropic API error %s (request_id=%s)", exc.status_code, exc.request_id)
        raise HTTPException(status_code=502, detail=f"LLM error: {exc.message}") from exc


@app.post("/api/v1/webhooks/lead", response_model=ChatResponse)
async def lead_webhook(request: ChatRequest) -> ChatResponse:
    """Event trigger for form fills / inbound emails; same pipeline as chat."""
    request.channel = "webhook" if request.channel == "chat" else request.channel
    return await chat(request)


@app.get("/api/v1/agent/sessions/{session_id}", response_model=ConversationThread)
async def get_session(session_id: str) -> ConversationThread:
    customer_id = get_store().customer_for_session(session_id)
    if customer_id is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    return await get_interaction_history(customer_id)


@app.get("/api/v1/crm/{customer_id}", response_model=CrmRecord)
async def get_crm_record(customer_id: str) -> CrmRecord:
    record = get_store().records.get(customer_id)
    if record is None:
        raise HTTPException(status_code=404, detail="No CRM record")
    return record
