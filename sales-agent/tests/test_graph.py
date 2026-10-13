from conftest import FEW_SHOT

from sales_agent.agent import SalesAgent
from sales_agent.graph import STAGES
from sales_agent.models import ChatRequest, CustomerRef, Intent, NextBestAction
from sales_agent.store import get_store


async def test_few_shot_competitor_negotiation():
    resp = await SalesAgent().handle(ChatRequest(message=FEW_SHOT))
    assert [t.stage for t in resp.trace] == list(STAGES)
    assert resp.intent.intent == Intent.NEGOTIATION
    assert resp.intent.entities.competitor_price == 20
    assert resp.strategy.action == NextBestAction.HANDLE_OBJECTION
    assert resp.quote and resp.quote.unit_price == 18.0 and resp.quote.quantity == 50
    assert "$18" in resp.reply and "CompetitorX" in resp.reply
    assert resp.guardrail_flags == []
    assert resp.meeting is not None
    assert get_store().records[resp.customer_id].stage == "negotiation"
    used = {c.tool for c in resp.tool_calls}
    assert {"search_product_catalog", "get_competitor_analysis", "calculate_dynamic_quote",
            "update_crm_stage"} <= used


async def test_known_lead_is_enriched_and_personalised():
    resp = await SalesAgent().handle(ChatRequest(
        message="We're evaluating options for tracking our delivery vans, can we see a demo?",
        customer=CustomerRef(email="mei@swiftmove-logistics.sg"),
    ))
    assert resp.customer_id == "cust-swiftmove"
    assert resp.reply.startswith("Hi Mei")
    assert resp.strategy.action == NextBestAction.PROPOSE_DEMO


async def test_b2c_research_is_educational():
    resp = await SalesAgent().handle(ChatRequest(
        message="Just curious how solar panels could lower the energy bill for my home in Australia.",
        customer=CustomerRef(email="alex@gmail.com"),
    ))
    assert resp.intent.entities.market == "B2C"
    assert resp.quote and resp.quote.product_id == "solarflex-panels"
    assert resp.strategy.action == NextBestAction.INFORM_EDUCATE
    assert resp.meeting is None


async def test_vague_message_triggers_discovery():
    resp = await SalesAgent().handle(ChatRequest(message="Hello, what do you sell?"))
    assert resp.strategy.action == NextBestAction.DISCOVERY
    assert resp.reply.rstrip().endswith("?")


async def test_session_keeps_history():
    agent = SalesAgent()
    first = await agent.handle(ChatRequest(message=FEW_SHOT, customer=CustomerRef(domain="swiftmove-logistics.sg")))
    second = await agent.handle(ChatRequest(message="Does it integrate with SAP?", session_id=first.session_id))
    assert second.customer_id == first.customer_id
    assert second.intent.intent == Intent.TECHNICAL_INQUIRY
    assert len(get_store().interactions[first.customer_id]) == 4
