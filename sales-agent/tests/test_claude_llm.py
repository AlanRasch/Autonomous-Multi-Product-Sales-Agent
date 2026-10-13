"""Exercises the Claude code path with a stubbed client (no network, no API key)."""

from types import SimpleNamespace

from conftest import FEW_SHOT

from sales_agent.agent import SalesAgent
from sales_agent.config import Settings
from sales_agent.llm.claude import ClaudeLLM
from sales_agent.llm.mock import MockLLM
from sales_agent.models import ChatRequest


def _text(t):
    return SimpleNamespace(type="text", text=t)


class FakeMessages:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    async def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        parsed = await MockLLM().classify(FEW_SHOT, [])
        return SimpleNamespace(stop_reason="end_turn", parsed_output=parsed)

    async def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        return self.replies.pop(0)


def _client(replies):
    messages = FakeMessages(replies)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


async def test_claude_drafting_loop_runs_tools_and_respects_guardrails():
    tool_turn = SimpleNamespace(
        stop_reason="tool_use",
        content=[
            _text("Let me verify pricing."),
            SimpleNamespace(type="tool_use", id="tu_1", name="calculate_dynamic_quote",
                            input={"product_id": "fleetpro-iot", "quantity": 50}),
        ],
    )
    final = SimpleNamespace(stop_reason="end_turn", content=[_text(
        "For 50 vehicles on an annual plan we can offer $18 per vehicle/month, below CompetitorX's $20. "
        "Open to a 15-minute preview Thursday?"
    )])
    client, messages = _client([tool_turn, final])
    settings = Settings(llm_mode="claude")
    agent = SalesAgent(llm=ClaudeLLM(settings, client=client), settings=settings)

    resp = await agent.handle(ChatRequest(message=FEW_SHOT))

    assert resp.llm_mode == "claude"
    assert "$18" in resp.reply
    assert resp.guardrail_flags == []
    exec_tools = [c.tool for c in resp.tool_calls if c.stage == "execution"]
    assert "calculate_dynamic_quote" in exec_tools
    # Request shape: model, effort, fallbacks, tools, and the tool result sent back in one user turn.
    kind, kwargs = messages.calls[1]
    assert kind == "create" and kwargs["model"] == "claude-opus-5-5"
    assert kwargs["fallbacks"] == "default" and kwargs["output_config"] == {"effort": "medium"}
    assert all(t["name"] != "update_crm_stage" for t in kwargs["tools"])
    second_request = messages.calls[2][1]["messages"]
    assert second_request[-1]["content"][0]["type"] == "tool_result"


async def test_claude_revises_reply_when_guardrail_trips():
    bad = SimpleNamespace(stop_reason="end_turn", content=[_text("We'll do $12 per vehicle, deal?")])
    good = SimpleNamespace(stop_reason="end_turn", content=[_text("Our authorized price is $18 per vehicle/month. Demo Thursday?")])
    client, _ = _client([bad, good])
    settings = Settings(llm_mode="claude")
    agent = SalesAgent(llm=ClaudeLLM(settings, client=client), settings=settings)

    resp = await agent.handle(ChatRequest(message=FEW_SHOT))
    assert "$18" in resp.reply and "$12" not in resp.reply
    assert resp.guardrail_flags == []
