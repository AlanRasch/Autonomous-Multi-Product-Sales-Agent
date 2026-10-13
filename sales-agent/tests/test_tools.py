import pytest

from sales_agent.guardrails import check_reply
from sales_agent.tools import run_tool
from sales_agent.tools.customer import enrich_lead_profile
from sales_agent.tools.execution import calculate_dynamic_quote, update_crm_stage
from sales_agent.tools.market import get_competitor_analysis
from sales_agent.tools.product import search_product_catalog


async def test_quote_matches_spec_example():
    q = await calculate_dynamic_quote("fleetpro-iot", 50)
    assert q.unit_price == 18.0
    assert q.discount_pct == 28
    assert q.total_per_period == 900.0
    assert not q.discount_capped
    assert q.margin_pct >= 35


async def test_quote_caps_discount_at_authorized_limit():
    q = await calculate_dynamic_quote("fleetpro-iot", 300, discount_code="PARTNER15")
    assert q.discount_capped
    assert q.discount_pct <= 32
    assert q.margin_pct >= 35  # margin floor holds
    assert any("approval" in n for n in q.notes)


async def test_invalid_discount_code_is_rejected():
    q = await calculate_dynamic_quote("cloudcrm-suite", 10, discount_code="FREE99", contract_term="monthly")
    assert q.rejected_discount_code == "FREE99"
    assert q.unit_price == 49.0


async def test_catalog_search_ranks_relevant_product_first():
    results = await search_product_catalog("real-time fleet tracking for trucks")
    assert results[0].product_id == "fleetpro-iot"
    b2c = await search_product_catalog("energy monitoring", {"market": "B2C"})
    assert all(p.market != "B2B" for p in b2c)


async def test_unknown_competitor_and_lead_are_marked():
    comp = await get_competitor_analysis("CRM", "NobodyCorp")
    assert comp.known is False
    lead = await enrich_lead_profile("jane@unknown-startup.com")
    assert lead.enriched is False
    known = await enrich_lead_profile("ops@swiftmove-logistics.sg")
    assert known.enriched and known.employee_count == 50


async def test_run_tool_validates_arguments():
    _, record = await run_tool("calculate_dynamic_quote", {"product_id": "fleetpro-iot", "quantity": 0}, "test")
    assert not record.ok
    _, record = await run_tool("no_such_tool", {}, "test")
    assert not record.ok
    with pytest.raises(ValueError):
        await update_crm_stage("c1", "bogus-stage")


async def test_guardrails():
    q = await calculate_dynamic_quote("fleetpro-iot", 50)
    evidence = [q.model_dump_json(), "CompetitorX list_price 20.0"]
    ok = "We can offer $18 per vehicle/month vs CompetitorX at $20. Want a demo?"
    assert check_reply(ok, evidence, q) == []
    bad = "We can do $15 per vehicle/month, that's 40% off! Deal."
    flags = check_reply(bad, evidence, q)
    assert any(f.startswith("unverified_figure") for f in flags)
    assert any(f.startswith("price_below_authorized") for f in flags)
    assert any(f.startswith("discount_exceeds_authorized") for f in flags)
    assert any(f.startswith("missing_cta") for f in flags)
