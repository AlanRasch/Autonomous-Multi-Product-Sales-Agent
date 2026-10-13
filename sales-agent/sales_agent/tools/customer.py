"""Customer & demographic intelligence tools.

`enrich_lead_profile` would normally call Clearbit/Apollo/ZoomInfo; here it reads
data/leads.json and returns a clearly-marked placeholder for unknown domains.
"""

from __future__ import annotations

import re

from .. import data_loader
from ..models import ConversationThread, LeadProfile, SegmentData
from ..store import get_store

FREE_EMAIL_DOMAINS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "proton.me",
    "protonmail.com", "aol.com", "live.com",
}


def domain_of(email_or_domain: str) -> str:
    value = email_or_domain.strip().lower()
    if "@" in value:
        value = value.split("@", 1)[1]
    value = re.sub(r"^https?://", "", value).split("/")[0]
    return value.removeprefix("www.")


def customer_id_for(email_or_domain: str) -> str:
    value = email_or_domain.strip().lower()
    domain = domain_of(value)
    if domain in FREE_EMAIL_DOMAINS:
        slug = re.sub(r"[^a-z0-9]+", "-", value.split("@")[0]).strip("-")
        return f"cust-{slug}"
    known = data_loader.leads().get(domain)
    if known:
        return known["customer_id"]
    return "cust-" + re.sub(r"[^a-z0-9]+", "-", domain.split(".")[0]).strip("-")


async def enrich_lead_profile(email_or_domain: str) -> LeadProfile:
    domain = domain_of(email_or_domain)
    known = data_loader.leads().get(domain)
    if known:
        return LeadProfile(**known)
    if domain in FREE_EMAIL_DOMAINS:
        return LeadProfile(
            domain=domain,
            company_name="Individual consumer",
            industry="consumer",
            customer_id=customer_id_for(email_or_domain),
            enriched=False,
        )
    return LeadProfile(
        domain=domain,
        company_name=domain.split(".")[0].replace("-", " ").title(),
        industry="unknown",
        customer_id=customer_id_for(email_or_domain),
        enriched=False,
    )


async def get_demographic_segment(geography: str, age_bracket: str = "all", income_tier: str = "all") -> SegmentData:
    segments = data_loader.intel()["segments"]
    geo = (geography or "").lower()
    data = next((v for k, v in segments.items() if k != "default" and (k in geo or geo in k)), None)
    data = data or segments["default"]
    ppi = data["purchasing_power_index"]
    if income_tier.lower() in ("high", "affluent"):
        ppi *= 1.3
    elif income_tier.lower() in ("low", "budget"):
        ppi *= 0.7
    return SegmentData(
        geography=geography or "Global",
        age_bracket=age_bracket,
        income_tier=income_tier,
        purchasing_power_index=round(ppi, 2),
        spending_trends=data["spending_trends"],
        cultural_preferences=data["cultural_preferences"],
    )


async def get_interaction_history(customer_id: str) -> ConversationThread:
    store = get_store()
    rec = store.records.get(customer_id)
    return ConversationThread(
        customer_id=customer_id,
        interactions=store.interactions.get(customer_id, [])[-20:],
        current_stage=rec.stage if rec else None,
    )
