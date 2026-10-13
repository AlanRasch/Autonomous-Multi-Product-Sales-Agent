"""Offline, deterministic LLM stand-in.

Lets the whole graph run (and be tested) with no API key: regex/keyword intent
extraction and template-based replies that only use facts from tool results.
"""

from __future__ import annotations

import re

from .. import data_loader
from ..models import (
    Entities,
    Intent,
    IngestionResult,
    Interaction,
    NextBestAction,
)
from .base import DraftContext, DraftResult

_REGIONS = {
    "southeast asia": "Southeast Asia", "se asia": "Southeast Asia", "singapore": "Southeast Asia",
    "indonesia": "Southeast Asia", "malaysia": "Southeast Asia", "thailand": "Southeast Asia",
    "vietnam": "Southeast Asia", "philippines": "Southeast Asia", "north america": "North America",
    "usa": "North America", "united states": "North America", "canada": "North America",
    "us-based": "North America", "europe": "Europe", "uk": "Europe", "germany": "Europe",
    "france": "Europe", "australia": "Australia",
}
_INDUSTRIES = {
    "logistics": "logistics", "fleet": "logistics", "trucking": "logistics", "delivery": "logistics",
    "transport": "logistics", "saas": "saas", "software": "saas", "startup": "saas",
    "clinic": "healthcare", "hospital": "healthcare", "healthcare": "healthcare",
    "retail": "retail", "store": "retail", "manufactur": "manufacturing", "factory": "manufacturing",
    "homeowner": "consumer", "my home": "consumer", "my house": "consumer", "household": "consumer",
}
_UNITS = r"vehicles?|trucks?|vans?|cars?|seats?|users?|licen[cs]es?|endpoints?|devices?|panels?|learners?|units?|hubs?"


def _money(x: float) -> str:
    return f"{x:g}" if float(x).is_integer() else f"{x:.2f}"


def _has(text: str, *words: str) -> bool:
    return any(w in text for w in words)


class MockLLM:
    mode = "mock"

    async def classify(self, message: str, history: list[Interaction]) -> IngestionResult:
        t = message.lower()
        intel = data_loader.intel()

        competitors = [c["competitor_name"] for c in intel["competitors"] if c["competitor_name"].lower() in t]
        prices = [float(p.replace(",", "")) for p in re.findall(r"\$\s?(\d[\d,]*(?:\.\d+)?)", message)]
        qty = re.search(rf"(\d[\d,]*)\s*(?:{_UNITS})\b", t)
        size = re.search(r"(\d[\d,]*)[- ]?(?:person|people|employee|staff|member)", t)
        region = next((v for k, v in _REGIONS.items() if re.search(rf"\b{re.escape(k)}\b", t)), None)
        industry = next((v for k, v in _INDUSTRIES.items() if k in t), None)
        timeline = re.search(
            r"(this|next) (week|month|quarter|year)|within \d+ (days?|weeks?|months?)|by (q[1-4]|end of \w+)|asap", t
        )
        budget = re.search(r"budget[^$\d.]{0,25}\$?\s?(\d[\d,]*k?)", t)
        budget_approved = bool(re.search(r"budget (is |has been )?(approved|allocated|signed off)", t))
        code = next((c for c in intel["discount_codes"] if c.lower() in t), None)

        objections: list[str] = []
        if _has(t, "expensive", "too much", "cheaper", "lower price", "can you match", "beat", "afford", "over budget"):
            objections.append("price")
        if competitors or _has(t, "competitor", "already use", "currently use", "switch from"):
            objections.append("competitor")
        if _has(t, "complicated", "complex", "hard to", "difficult", "learning curve"):
            objections.append("complexity")
        if _has(t, "not now", "next year", "later", "not the right time", "on hold"):
            objections.append("timing")
        if _has(t, "trust", "not sure", "skeptical", "guarantee"):
            objections.append("trust")

        if _has(t, "match", "discount", "quote", "pricing", "price", "deal", "negotiat") or prices:
            intent = Intent.NEGOTIATION
        elif objections and not prices:
            intent = Intent.OBJECTION
        elif _has(t, "integrat", "api", "how does", "security", "compatible", "spec", "support for", "does it work"):
            intent = Intent.TECHNICAL_INQUIRY
        elif _has(t, "re:", "following up", "you reached out", "your email", "got your message"):
            intent = Intent.COLD_OUTREACH_RESPONSE
        else:
            intent = Intent.INBOUND_LEAD

        if _has(t, "quote", "contract", "sign", "match", "purchase", "buy", "order", "start date", "invoice"):
            readiness = "ready_to_buy"
        elif _has(t, "compare", "demo", "trial", "how does", "evaluat", "looking for", "options") or competitors:
            readiness = "evaluating"
        else:
            readiness = "research"

        pain = [p for p in ("cost", "fuel", "safety", "visibility", "latency", "ransomware", "compliance",
                            "forecast", "turnover", "energy bill", "downtime") if p in t]
        outcomes = [o for o in ("real-time tracking", "reduce costs", "save money", "grow", "automate",
                                "faster", "protect", "visibility") if o in t]
        b2c = industry == "consumer" or _has(t, "my home", "my house", "my family", "for myself")

        secondary = [Intent.OBJECTION] if objections and intent != Intent.OBJECTION else []
        entities = Entities(
            industry=industry,
            company_size=int(size.group(1).replace(",", "")) if size else None,
            location=region,
            budget=budget.group(0) if budget else ("approved" if budget_approved else None),
            budget_amount=float(budget.group(1).replace(",", "").replace("k", "000")) if budget else None,
            timeline=timeline.group(0) if timeline else None,
            quantity=int(qty.group(1).replace(",", "")) if qty else None,
            desired_outcomes=outcomes,
            pain_points=pain,
            product_interest=message[:200],
            competitors_mentioned=competitors,
            competitor_price=prices[0] if competitors and prices else None,
            discount_code=code,
            contract_term="annual" if _has(t, "annual", "yearly", "12-month", "12 month") else
                          ("monthly" if _has(t, "month-to-month", "monthly contract", "no contract") else None),
            is_decision_maker=True if re.search(r"\bi'?m (the )?(ceo|owner|founder|director|head|vp|cfo|coo|cto)\b|\bwe run\b|\bi run\b|\bi own\b", t) else None,
            market="B2C" if b2c else ("B2B" if industry or size else None),
        )
        return IngestionResult(
            intent=intent,
            secondary_intents=secondary,
            purchase_readiness=readiness,  # type: ignore[arg-type]
            objection_types=objections,  # type: ignore[arg-type]
            entities=entities,
            summary=f"{intent.value.replace('_', ' ').title()} ({readiness}); "
                    f"{industry or 'unknown industry'}, {region or 'unknown region'}.",
        )

    async def draft(self, ctx: DraftContext) -> DraftResult:
        e, a, s = ctx.enrichment, ctx.analysis, ctx.strategy
        spec, quote = e.primary_product, a.quote
        ents = ctx.ingestion.entities
        lead_name = None
        if e.lead and e.lead.decision_makers:
            lead_name = e.lead.decision_makers[0].name.split()[0]
        parts: list[str] = [f"Hi {lead_name}," if lead_name else "Hi there,"]

        industry_pain = e.industry.pain_points[0].lower() if e.industry and e.industry.pain_points else None
        region = ents.location or (e.lead.region if e.lead else None)
        if spec:
            framing = f"Thanks for reaching out! {spec.category} decisions"
            if region:
                framing += f" in {region}"
            framing += " come down to outcomes and predictable cost"
            if industry_pain:
                who = "households" if e.industry.industry == "consumer" else f"{e.industry.industry} teams"  # type: ignore[union-attr]
                framing += f", especially with {industry_pain} top of mind for most {who}"
            parts.append(framing + ".")
        else:
            parts.append("Thanks for reaching out! I want to make sure I point you to the right solution.")

        comp = next((c for c in e.competitors if c.known), None)
        if comp and s.action in (NextBestAction.HANDLE_OBJECTION, NextBestAction.PROPOSE_QUOTE, NextBestAction.CLOSE):
            line = f"While {comp.competitor_name}"
            if comp.list_price is not None:
                line += f" offers a ${_money(comp.list_price)}/{comp.price_unit} baseline"
            if comp.weaknesses:
                line += ", their plan comes with trade-offs: " + "; ".join(w[0].lower() + w[1:] for w in comp.weaknesses[:2])
            parts.append(line + ".")

        if e.persona_pitch and s.action == NextBestAction.HANDLE_OBJECTION:
            reframes = [e.persona_pitch.common_objections[o] for o in ctx.ingestion.objection_types
                        if o in e.persona_pitch.common_objections and o not in ("price", "competitor")]
            parts.extend(reframes[:1])

        if spec and s.action != NextBestAction.DISCOVERY:
            points = [f"{i}. {vm.capability}: {vm.outcome}" for i, vm in enumerate(a.value_map[:3], 1)]
            if not points and e.persona_pitch:
                if s.action == NextBestAction.INFORM_EDUCATE:
                    parts.append(e.persona_pitch.headline)
                points = [f"{i}. {o}" for i, o in enumerate(e.persona_pitch.outcomes[:3], 1)]
            price_talk = s.action in (NextBestAction.PROPOSE_QUOTE, NextBestAction.CLOSE) or (
                s.action == NextBestAction.HANDLE_OBJECTION
                and {"price", "competitor"} & set(ctx.ingestion.objection_types))
            if quote and price_talk:
                unit_noun = quote.unit.split("/")[0]
                price_line = (
                    f"{len(points) + 1}. Pricing: for {quote.quantity} {unit_noun}s on an {quote.contract_term} "
                    f"agreement, we can offer ${_money(quote.unit_price)} per {quote.unit}"
                    if quote.contract_term == "annual" else
                    f"{len(points) + 1}. Pricing: for {quote.quantity} {unit_noun}s month-to-month, "
                    f"we can offer ${_money(quote.unit_price)} per {quote.unit}"
                )
                if ents.competitor_price and comp and quote.unit_price < ents.competitor_price:
                    price_line += f", below {comp.competitor_name}'s quote while including more"
                elif ents.competitor_price and comp and comp.hidden_costs:
                    price_line += f"; once {comp.competitor_name}'s add-ons ({comp.hidden_costs[0]}) are counted, total cost of ownership favors us"
                points.append(price_line + ".")
            if points:
                parts.append(f"With {spec.name}:\n" + "\n".join(points))
            if spec.social_proof and s.action != NextBestAction.PROPOSE_QUOTE:
                parts.append(spec.social_proof[0] + ".")

        cta = s.cta
        if ctx.meeting and "{booking_url}" in cta:
            cta = cta.replace("{booking_url}", ctx.meeting.booking_url)
        parts.append(cta)
        return DraftResult(text="\n\n".join(parts))
