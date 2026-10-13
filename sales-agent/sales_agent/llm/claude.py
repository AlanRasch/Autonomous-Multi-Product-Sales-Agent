"""Claude-backed LLM: structured intent extraction + a tool-using drafting loop."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import anthropic

from ..config import Settings
from ..models import CalendarLink, IngestionResult, Interaction, Quote
from ..prompts import DRAFTING_INSTRUCTIONS, FEW_SHOT_EXAMPLE, INGESTION_INSTRUCTIONS, SYSTEM_PROMPT
from ..tools import DRAFTING_TOOLS, claude_tools, result_to_text, run_tool
from .base import DraftContext, DraftResult
from .mock import MockLLM

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _history_text(history: list[Interaction]) -> str:
    if not history:
        return "(no prior interactions)"
    return "\n".join(f"- {i.timestamp:%Y-%m-%d} {i.direction} via {i.channel}: {i.summary}" for i in history[-10:])


class ClaudeLLM:
    mode = "claude"

    def __init__(self, settings: Settings, client: anthropic.AsyncAnthropic | None = None) -> None:
        self.settings = settings
        self.client = client or anthropic.AsyncAnthropic()
        self._heuristic = MockLLM()  # used only if the model refuses or returns nothing usable

    def _common(self, effort: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": self.settings.max_tokens,
            "output_config": {"effort": effort},
        }
        if self.settings.enable_fallbacks:
            # Re-run on a fallback model if a safety classifier declines (Claude API only).
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    # -- STATE 1: ingestion ------------------------------------------------
    async def classify(self, message: str, history: list[Interaction]) -> IngestionResult:
        response = await self.client.beta.messages.parse(
            **self._common("low"),
            system=f"{SYSTEM_PROMPT}\n\n{INGESTION_INSTRUCTIONS}",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<history>\n{_history_text(history)}\n</history>\n\n"
                        f"<customer_message>\n{message}\n</customer_message>"
                    ),
                }
            ],
            output_format=IngestionResult,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            log.warning("Ingestion fell back to heuristics (stop_reason=%s)", response.stop_reason)
            return await self._heuristic.classify(message, history)
        return response.parsed_output

    # -- STATE 5: drafting with tool use ------------------------------------
    async def draft(self, ctx: DraftContext) -> DraftResult:
        e = ctx.enrichment
        context = {
            "intent": ctx.ingestion.model_dump(mode="json"),
            "lead": e.lead.model_dump(mode="json") if e.lead else None,
            "primary_product": e.primary_product.model_dump(mode="json") if e.primary_product else None,
            "other_matching_products": [p.model_dump(mode="json") for p in e.products[1:3]],
            "persona_pitch": e.persona_pitch.model_dump(mode="json") if e.persona_pitch else None,
            "competitors": [c.model_dump(mode="json") for c in e.competitors],
            "industry": e.industry.model_dump(mode="json") if e.industry else None,
            "market_signals": e.market.model_dump(mode="json") if e.market else None,
            "segment": e.segment.model_dump(mode="json") if e.segment else None,
            "analysis": ctx.analysis.model_dump(mode="json"),
            "strategy": ctx.strategy.model_dump(mode="json"),
            "meeting": ctx.meeting.model_dump(mode="json") if ctx.meeting else None,
        }
        user_content = (
            f"<customer_id>{ctx.customer_id}</customer_id>\n"
            f"<history>\n{_history_text(ctx.history)}\n</history>\n"
            f"<context>\n{json.dumps(context, default=str)}\n</context>\n\n"
            f"<customer_message>\n{ctx.message}\n</customer_message>"
        )
        if ctx.feedback:
            user_content += f"\n\n<guardrail_feedback>\n{ctx.feedback}\n</guardrail_feedback>"

        system = [
            {"type": "text", "text": f"{SYSTEM_PROMPT}\n\n{FEW_SHOT_EXAMPLE}\n\n{DRAFTING_INSTRUCTIONS}",
             "cache_control": {"type": "ephemeral"}},
        ]
        tools = claude_tools(DRAFTING_TOOLS)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        result = DraftResult(text="")

        for _ in range(self.settings.max_tool_iterations):
            response = await self.client.beta.messages.create(
                **self._common(self.settings.effort), system=system, tools=tools, messages=messages
            )
            if response.stop_reason == "refusal":
                log.warning("Drafting refused; using template reply")
                return await self._heuristic.draft(ctx)
            if response.stop_reason == "pause_turn":
                messages.append({"role": "assistant", "content": response.content})
                continue

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                result.text = "\n".join(b.text for b in response.content if b.type == "text").strip()
                break

            messages.append({"role": "assistant", "content": response.content})
            outcomes = await asyncio.gather(
                *(run_tool(b.name, dict(b.input), stage="execution") for b in tool_uses)
            )
            tool_results = []
            for block, (value, record) in zip(tool_uses, outcomes):
                text = result_to_text(value, record)
                result.tool_calls.append(record)
                result.tool_outputs.append(f"{block.name}: {text}")
                if isinstance(value, Quote):
                    result.quote = value
                elif isinstance(value, CalendarLink):
                    result.meeting = value
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": text, "is_error": not record.ok}
                )
            # All results for parallel calls go back in ONE user message.
            messages.append({"role": "user", "content": tool_results})
        else:
            log.warning("Drafting hit max_tool_iterations without a final answer")

        if not result.text:
            fallback = await self._heuristic.draft(ctx)
            result.text = fallback.text
        return result
