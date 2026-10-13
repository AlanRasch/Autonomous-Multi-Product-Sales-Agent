"""The five state-machine stages: Ingestion -> Enrichment -> Analysis -> Strategy -> Execution."""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any

from ..guardrails import check_reply
from ..llm import DraftContext, SalesLLM
from ..models import (
    AnalysisResult,
    BantScore,
    CalendarLink,
    CompetitorMatrix,
    ConversationThread,
    EnrichmentBundle,
    IndustryReport,
    Interaction,
    LeadProfile,
    NextBestAction,
    PersonaPitch,
    Product,
    ProductSpec,
    Quote,
    SegmentData,
    StageTrace,
    StrategyDecision,
    ToolCallRecord,
    ValueMapping,
    WebSearchResults,
)
from ..store import get_store
from ..tools import result_to_text, run_tool
from ..tools.customer import customer_id_for
from ..tools.market import normalize_industry
from .state import AgentState


class _ToolSession:
    """Runs tools for one stage and collects audit records + evidence text."""

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.records: list[ToolCallRecord] = []
        self.evidence: list[str] = []
        self.errors: list[str] = []

    async def call(self, name: str, **arguments: Any) -> Any:
        result, record = await run_tool(name, arguments, self.stage)
        self.records.append(record)
        if record.ok:
            self.evidence.append(f"{name}: {result_to_text(result, record)}")
        else:
            self.errors.append(f"{name}: {record.error}")
        return result

    def update(self, started: float, summary: str, **values: Any) -> dict[str, Any]:
        return {
            **values,
            "tool_calls": self.records,
            "evidence": self.evidence,
            "trace": [StageTrace(stage=self.stage, duration_ms=round((time.perf_counter() - started) * 1000, 1), summary=summary)],
        }


def _tokens(text: str) -> set[str]:
    # Crude stemming (6-char prefix) so "forecast" matches "forecasting".
    return {t[:6] for t in re.findall(r"[a-z]+", text.lower()) if len(t) > 3}


# ---------------------------------------------------------------------------
# STATE 1: INGESTION - parse intent & extract entities
# ---------------------------------------------------------------------------


def make_ingestion(llm: SalesLLM):
    async def ingestion(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        req = state["request"]
        store = get_store()
        ref = req.customer
        customer_id = (
            ref.customer_id
            or (customer_id_for(ref.email) if ref.email else None)
            or (customer_id_for(ref.domain) if ref.domain else None)
            or store.customer_for_session(state["session_id"])
            or f"anon-{state['session_id'][:8]}"
        )
        store.bind_session(state["session_id"], customer_id)
        history = store.interactions.get(customer_id, [])
        result = await llm.classify(req.message, history)
        return {
            "customer_id": customer_id,
            "history": history,
            "ingestion": result,
            "trace": [StageTrace(stage="ingestion", duration_ms=round((time.perf_counter() - started) * 1000, 1),
                                 summary=f"intent={result.intent.value}, readiness={result.purchase_readiness}")],
        }

    return ingestion


# ---------------------------------------------------------------------------
# STATE 2: ENRICHMENT - product + customer + market data, in parallel
# ---------------------------------------------------------------------------


def _persona(lead: LeadProfile | None, market: str | None) -> str:
    if market == "B2C" or (lead and lead.industry == "consumer"):
        return "homeowner"
    title = (lead.decision_makers[0].title.lower() if lead and lead.decision_makers else "")
    if any(k in title for k in ("sales", "revops", "revenue")):
        return "sales_leader"
    if any(k in title for k in ("cio", "cto", "ciso", "it ", "security")):
        return "it_leader"
    if any(k in title for k in ("ceo", "cfo", "founder", "owner", "president")):
        return "executive"
    return "operations"


async def enrichment(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    tools = _ToolSession("enrichment")
    req, ing = state["request"], state["ingestion"]
    ents = ing.entities
    lead_key = req.customer.email or req.customer.domain

    filters = {"market": ents.market} if ents.market else None
    query = ents.product_interest or req.message
    phase1 = [
        tools.call("search_product_catalog", query=query, filters=filters),
        tools.call("get_interaction_history", customer_id=state["customer_id"]),
        tools.call("enrich_lead_profile", email_or_domain=lead_key) if lead_key else asyncio.sleep(0, None),
    ]
    if ents.location or ents.market == "B2C":
        phase1.append(tools.call(
            "get_demographic_segment",
            geography=ents.location or "Global",
            age_bracket=ents.age_bracket or "all",
            income_tier=ents.income_tier or "all",
        ))
    products, history, lead, *rest = await asyncio.gather(*phase1)
    segment: SegmentData | None = rest[0] if rest else None
    products = products or []

    top: Product | None = products[0] if products else None
    industry = normalize_industry(ents.industry or (lead.industry if lead and lead.enriched else None))
    region = ents.location or (lead.region if lead else None) or "Global"

    phase2: dict[str, Any] = {}
    if top:
        phase2["spec"] = tools.call("get_product_details", product_id=top.product_id)
        phase2["pitch"] = tools.call("get_value_propositions", product_id=top.product_id,
                                     target_persona=_persona(lead, ents.market))
        for i, comp in enumerate(ents.competitors_mentioned):
            phase2[f"comp{i}"] = tools.call("get_competitor_analysis", product_category=top.category,
                                            competitor_name=comp)
    if industry != "unknown":
        phase2["industry"] = tools.call("get_industry_benchmarks", industry=industry)
    market_query = " ".join(filter(None, [top.category if top else None, industry if industry != "unknown" else None])) or query
    phase2["market"] = tools.call("search_web_market_data", query=market_query, geography=region)
    values = dict(zip(phase2, await asyncio.gather(*phase2.values())))

    bundle = EnrichmentBundle(
        products=products,
        primary_product=values.get("spec") if isinstance(values.get("spec"), ProductSpec) else None,
        persona_pitch=values.get("pitch") if isinstance(values.get("pitch"), PersonaPitch) else None,
        lead=lead if isinstance(lead, LeadProfile) else None,
        history=history if isinstance(history, ConversationThread) else None,
        industry=values.get("industry") if isinstance(values.get("industry"), IndustryReport) else None,
        competitors=[v for k, v in values.items() if k.startswith("comp") and isinstance(v, CompetitorMatrix)],
        market=values.get("market") if isinstance(values.get("market"), WebSearchResults) else None,
        segment=segment,
        errors=tools.errors,
    )
    summary = (f"{len(products)} product match(es); top={top.name if top else 'none'}; "
               f"lead={'enriched' if bundle.lead and bundle.lead.enriched else 'unknown'}; "
               f"competitors={len(bundle.competitors)}")
    return tools.update(started, summary, enrichment=bundle)


# ---------------------------------------------------------------------------
# STATE 3: ANALYSIS - BANT qualification, value mapping, margin check
# ---------------------------------------------------------------------------


def _bant(state: AgentState, top_score: float) -> BantScore:
    ing, bundle = state["ingestion"], state["enrichment"]
    ents = ing.entities
    lead = bundle.lead
    budget = 25 if (ents.budget_amount or ents.budget or ents.competitor_price or ing.purchase_readiness == "ready_to_buy") else (
        12 if (ents.company_size or (lead and lead.employee_count)) else 5)
    authority = 25 if ents.is_decision_maker else (15 if lead and lead.decision_makers else 8)
    need = 25 if top_score >= 0.5 else (15 if top_score >= 0.25 else 5)
    near = bool(ents.timeline and re.search(r"this|asap|within|next (week|month)", ents.timeline))
    timeline = 25 if near else (15 if ents.timeline or ing.purchase_readiness == "ready_to_buy" else
                                (10 if ing.purchase_readiness == "evaluating" else 5))
    return BantScore(budget=budget, authority=authority, need=need, timeline=timeline)


def _value_map(state: AgentState) -> list[ValueMapping]:
    bundle, ents = state["enrichment"], state["ingestion"].entities
    spec, pitch = bundle.primary_product, bundle.persona_pitch
    if not spec:
        return []
    pains = list(dict.fromkeys(ents.pain_points + (bundle.industry.pain_points[:3] if bundle.industry else [])))
    outcomes = list(pitch.outcomes) if pitch else []
    used_features: set[str] = set()
    mapping: list[ValueMapping] = []
    for pain in pains:
        pt = _tokens(pain)
        feature = max((f for f in spec.features if f not in used_features),
                      key=lambda f: len(pt & _tokens(f)), default=None)
        if feature is None or not (pt & _tokens(feature)):
            continue
        used_features.add(feature)
        outcome = max(outcomes, key=lambda o: len((pt | _tokens(feature)) & _tokens(o)), default=None)
        if outcome:
            outcomes.remove(outcome)
        mapping.append(ValueMapping(pain_point=pain, capability=feature, outcome=outcome or f"Directly addresses {pain.lower()}"))
    return mapping[:4]


async def analysis(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    tools = _ToolSession("analysis")
    ing, bundle = state["ingestion"], state["enrichment"]
    ents, spec, lead = ing.entities, bundle.primary_product, bundle.lead
    top_score = (bundle.products[0].match_score or 0.0) if bundle.products else 0.0

    bant = _bant(state, top_score)
    total = bant.total
    qualification = ("sales_ready" if total >= 75 else "qualified" if total >= 55 else
                     "nurture" if total >= 35 else "unqualified")

    fit = top_score
    region = ents.location or (lead.region if lead else None)
    industry = bundle.industry.industry if bundle.industry else None
    if spec and industry and industry in spec.target_industries:
        fit += 0.1
    if spec and region and (region in spec.regions or "Global" in spec.regions):
        fit += 0.1

    gaps: list[str] = []
    quote: Quote | None = None
    if spec:
        unit_noun = spec.unit.split("/")[0]
        quantity = ents.quantity
        if quantity is None:
            if spec.market == "B2C":
                quantity = 1
            else:
                quantity = ents.company_size or (lead.employee_count if lead else None) or 1
                gaps.append(f"How many {unit_noun}s need coverage? (quote assumes {quantity})")
        quote = await tools.call(
            "calculate_dynamic_quote",
            product_id=spec.product_id,
            quantity=quantity,
            discount_code=ents.discount_code,
            contract_term=ents.contract_term or "annual",
        )
    else:
        gaps.append("What problem are you trying to solve, and for how many people or assets?")
    if bant.budget < 20:
        gaps.append("What budget range have you set aside?")
    if bant.timeline < 15:
        gaps.append("When do you need this live?")
    if bant.authority < 20:
        gaps.append("Who else is involved in the decision?")

    margin_ok = bool(quote and spec and quote.margin_pct >= spec.pricing.min_margin_pct) if quote else True
    can_beat = (quote.unit_price <= ents.competitor_price) if (quote and ents.competitor_price) else None

    result = AnalysisResult(
        bant=bant,
        bant_total=total,
        qualification=qualification,  # type: ignore[arg-type]
        fit_score=round(min(1.0, fit), 2),
        value_map=_value_map(state),
        quote=quote if isinstance(quote, Quote) else None,
        margin_ok=margin_ok,
        can_beat_competitor=can_beat,
        gaps=gaps,
    )
    summary = f"BANT={total}/100 ({qualification}); fit={result.fit_score}; margin_ok={margin_ok}"
    if result.quote:
        summary += f"; quote=${result.quote.unit_price:g}/{result.quote.unit} x{result.quote.quantity}"
    return tools.update(started, summary, analysis=result, quote=result.quote)


# ---------------------------------------------------------------------------
# STATE 4: STRATEGY - choose next best action, angle and CTA
# ---------------------------------------------------------------------------

_DEMO_CTA = "Would you be open to a {duration}-minute live demo on {slot_day}? You can also pick any time here: {booking_url}"


async def strategy(state: AgentState) -> dict[str, Any]:
    started = time.perf_counter()
    ing, bundle, a = state["ingestion"], state["enrichment"], state["analysis"]
    spec = bundle.primary_product
    objections = set(ing.objection_types)
    readiness = ing.purchase_readiness
    qualified = a.qualification in ("qualified", "sales_ready")
    existing_customer = bool(bundle.history and bundle.history.current_stage == "closed_won")

    talking = [f"{vm.capability} -> {vm.outcome}" for vm in a.value_map]
    for comp in bundle.competitors:
        talking += [f"{comp.competitor_name}: {w}" for w in comp.weaknesses[:2]]
    if a.quote:
        talking.append(f"Authorized price: ${a.quote.unit_price:g}/{a.quote.unit} ({a.quote.discount_pct:g}% off list)")

    if not spec or a.fit_score < 0.3:
        decision = StrategyDecision(
            action=NextBestAction.DISCOVERY, angle="consultative",
            cta="To point you to the right fit: " + (a.gaps[0] if a.gaps else "what are you looking to achieve?"),
            talking_points=a.gaps, crm_stage="discovery", follow_up_days=3,
            rationale="No confident product match yet - discover needs before pitching.",
        )
    elif objections & {"price", "competitor"}:
        pilot = " with a small pilot" if spec.market == "B2B" else ""
        decision = StrategyDecision(
            action=NextBestAction.HANDLE_OBJECTION,
            angle="competitive" if "competitor" in objections else "value_roi",
            cta=_DEMO_CTA.replace("live demo", f"live preview{pilot}"),
            talking_points=talking,
            crm_stage="negotiation" if readiness == "ready_to_buy" else "qualified",
            follow_up_days=2,
            rationale="Price/competitor pushback: reframe on TCO and differentiators, state only the authorized quote.",
        )
    elif objections:
        decision = StrategyDecision(
            action=NextBestAction.HANDLE_OBJECTION, angle="consultative",
            cta=_DEMO_CTA, talking_points=talking,
            crm_stage="qualified" if qualified else "nurture", follow_up_days=5,
            rationale=f"Non-price objection(s) {sorted(objections)}: de-risk with proof and a low-commitment next step.",
        )
    elif readiness == "ready_to_buy" and qualified:
        close = a.qualification == "sales_ready" and ing.intent.value == "negotiation"
        decision = StrategyDecision(
            action=NextBestAction.CLOSE if close else NextBestAction.PROPOSE_QUOTE,
            angle="upsell" if existing_customer else "close",
            cta="Shall I send the formal quote so you can review it with your team? Or book a 30-minute walkthrough here: {booking_url}",
            talking_points=talking, crm_stage="proposal_sent", follow_up_days=2,
            rationale="Strong purchase signals and qualified on BANT - present the authorized quote.",
        )
    elif readiness in ("evaluating", "ready_to_buy"):
        decision = StrategyDecision(
            action=NextBestAction.PROPOSE_DEMO, angle="upsell" if existing_customer else "value_roi",
            cta=_DEMO_CTA, talking_points=talking,
            crm_stage="qualified" if qualified else "discovery", follow_up_days=3,
            rationale="Actively evaluating - show value live before discussing price.",
        )
    else:
        decision = StrategyDecision(
            action=NextBestAction.INFORM_EDUCATE, angle="educational",
            cta=("Would a quick personalized estimate for your home be helpful? I can send one over."
                 if spec.market == "B2C" else
                 "Would a short overview of how teams like yours use it be helpful? I can send one over."),
            talking_points=talking, crm_stage="contacted" if qualified else "nurture", follow_up_days=7,
            rationale="Early-stage research - educate and build trust before selling.",
        )
    return {
        "strategy": decision,
        "trace": [StageTrace(stage="strategy", duration_ms=round((time.perf_counter() - started) * 1000, 1),
                             summary=f"action={decision.action.value}, angle={decision.angle}, stage={decision.crm_stage}")],
    }


# ---------------------------------------------------------------------------
# STATE 5: EXECUTION - draft reply, enforce guardrails, sync CRM
# ---------------------------------------------------------------------------

_MEETING_ACTIONS = {NextBestAction.PROPOSE_DEMO, NextBestAction.PROPOSE_QUOTE, NextBestAction.CLOSE,
                    NextBestAction.HANDLE_OBJECTION}


def make_execution(llm: SalesLLM):
    async def execution(state: AgentState) -> dict[str, Any]:
        started = time.perf_counter()
        tools = _ToolSession("execution")
        req, ing, a = state["request"], state["ingestion"], state["analysis"]
        decision = state["strategy"]
        customer_id = state["customer_id"]

        meeting: CalendarLink | None = None
        if decision.action in _MEETING_ACTIONS:
            duration = 30 if decision.action in (NextBestAction.PROPOSE_QUOTE, NextBestAction.CLOSE) else 15
            meeting = await tools.call("schedule_meeting", customer_id=customer_id, duration_mins=duration)
        cta = decision.cta
        if meeting:
            cta = cta.format(duration=meeting.duration_mins, slot_day=f"{meeting.proposed_slots[0]:%A}",
                             booking_url=meeting.booking_url)
        decision = decision.model_copy(update={"cta": cta})

        ctx = DraftContext(
            message=req.message, customer_id=customer_id, history=state.get("history", []),
            ingestion=ing, enrichment=state["enrichment"], analysis=a, strategy=decision, meeting=meeting,
        )
        draft = await llm.draft(ctx)
        evidence = [*state.get("evidence", []), *tools.evidence, req.message, cta]
        quote = draft.quote or a.quote
        flags = check_reply(draft.text, evidence + draft.tool_outputs, quote)
        if flags and llm.mode != "mock":
            ctx.feedback = "Revise the reply to fix these compliance issues:\n- " + "\n- ".join(flags)
            revised = await llm.draft(ctx)
            draft.tool_calls += revised.tool_calls
            draft.tool_outputs += revised.tool_outputs
            draft.text = revised.text
            quote = revised.quote or quote
            flags = check_reply(draft.text, evidence + draft.tool_outputs, quote)
        tools.records += draft.tool_calls

        sentiment = "negative" if ing.objection_types and ing.purchase_readiness == "research" else (
            "positive" if ing.purchase_readiness == "ready_to_buy" else "neutral")
        note = (f"{ing.intent.value} | {decision.action.value} | BANT {a.bant_total} | "
                + (f"quote {quote.quote_id} ${quote.unit_price:g}/{quote.unit} x{quote.quantity} | " if quote else "")
                + ing.summary)
        await tools.call("update_crm_stage", customer_id=customer_id, stage=decision.crm_stage, notes=note,
                         sentiment=sentiment, follow_up_days=decision.follow_up_days)

        store = get_store()
        now = datetime.now(timezone.utc)
        store.log_interaction(customer_id, Interaction(timestamp=now, channel=req.channel, direction="inbound",
                                                       summary=req.message[:500], sentiment=sentiment))  # type: ignore[arg-type]
        store.log_interaction(customer_id, Interaction(timestamp=now, channel=req.channel, direction="outbound",
                                                       summary=draft.text[:500]))  # type: ignore[arg-type]

        summary = f"reply={len(draft.text)} chars; tools={len(tools.records)}; flags={len(flags)}"
        return tools.update(
            started, summary,
            strategy=decision, reply=draft.text, quote=quote, meeting=draft.meeting or meeting,
            crm=store.records.get(customer_id), guardrail_flags=flags,
        )

    return execution
