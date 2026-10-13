"""Decision & execution tools: pricing, scheduling and CRM sync.

`calculate_dynamic_quote` is the single source of truth for discount authority - the
agent may never state a price lower than the quote's `min_authorized_unit_price`.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta, timezone

from .. import data_loader
from ..config import get_settings
from ..models import CalendarLink, Quote
from ..store import get_store
from .product import get_product_details

CRM_STAGES = (
    "new", "contacted", "discovery", "qualified", "demo_scheduled",
    "proposal_sent", "negotiation", "closed_won", "closed_lost", "nurture",
)


async def calculate_dynamic_quote(
    product_id: str,
    quantity: int,
    discount_code: str | None = None,
    contract_term: str = "annual",
) -> Quote:
    if quantity < 1:
        raise ValueError("quantity must be >= 1")
    if contract_term not in ("monthly", "annual"):
        raise ValueError("contract_term must be 'monthly' or 'annual'")
    spec = await get_product_details(product_id)
    rules = spec.pricing
    notes: list[str] = []

    breakdown: dict[str, float] = {}
    tier = max((t for t in rules.volume_tiers if quantity >= t.min_quantity), key=lambda t: t.min_quantity, default=None)
    if tier:
        breakdown[f"volume_{tier.min_quantity}+"] = tier.discount_pct
    if contract_term == "annual" and rules.annual_commit_discount_pct:
        breakdown["annual_commitment"] = rules.annual_commit_discount_pct

    rejected_code = None
    if discount_code:
        code_pct = data_loader.intel()["discount_codes"].get(discount_code.strip().upper())
        if code_pct is None:
            rejected_code = discount_code
            notes.append(f"Discount code '{discount_code}' is not valid.")
        else:
            breakdown[f"code_{discount_code.strip().upper()}"] = float(code_pct)

    requested = sum(breakdown.values())
    # Margin floor: price p must satisfy (p - cost) / p >= min_margin.
    floor_price = rules.unit_cost / (1 - rules.min_margin_pct / 100)
    margin_cap_pct = max(0.0, (1 - floor_price / rules.base_unit_price) * 100)
    authorized = min(rules.max_discount_pct, margin_cap_pct)
    applied = min(requested, authorized)
    capped = requested > authorized + 1e-9
    if capped:
        notes.append(
            f"Requested discounts total {requested:.0f}%; capped at the authorized {applied:.1f}%. "
            "Any further concession needs sales-manager approval."
        )

    unit_price = round(rules.base_unit_price * (1 - applied / 100), 2)
    margin_pct = round((unit_price - rules.unit_cost) / unit_price * 100, 1)
    if contract_term == "monthly" and rules.annual_commit_discount_pct:
        notes.append(
            f"An annual commitment adds up to {rules.annual_commit_discount_pct:.0f}% more discount "
            "(within authorized limits) - re-quote with contract_term='annual'."
        )

    return Quote(
        quote_id=f"Q-{uuid.uuid4().hex[:8].upper()}",
        product_id=product_id,
        product_name=spec.name,
        quantity=quantity,
        contract_term=contract_term,  # type: ignore[arg-type]
        currency=rules.currency,
        list_unit_price=rules.base_unit_price,
        unit_price=unit_price,
        unit=rules.unit,
        discount_pct=round(applied, 2),
        discount_breakdown=breakdown,
        total_per_period=round(unit_price * quantity, 2),
        margin_pct=margin_pct,
        min_authorized_unit_price=unit_price,
        discount_capped=capped,
        rejected_discount_code=rejected_code,
        valid_until=date.today() + timedelta(days=30),
        notes=notes,
    )


async def schedule_meeting(customer_id: str, duration_mins: int = 30) -> CalendarLink:
    """Propose the next three business-day slots at 10:00/14:00 UTC and return a booking link."""
    now = datetime.now(timezone.utc)
    slots: list[datetime] = []
    day = now.date() + timedelta(days=1)
    while len(slots) < 3:
        if day.weekday() < 5:
            for hour in (10, 14):
                if len(slots) < 3:
                    slots.append(datetime.combine(day, time(hour), tzinfo=timezone.utc))
        day += timedelta(days=1)
    base = get_settings().booking_base_url.rstrip("/")
    return CalendarLink(
        customer_id=customer_id,
        duration_mins=duration_mins,
        booking_url=f"{base}/{customer_id}?duration={duration_mins}",
        proposed_slots=slots,
    )


async def update_crm_stage(
    customer_id: str,
    stage: str,
    notes: str = "",
    sentiment: str | None = None,
    follow_up_days: int | None = None,
) -> bool:
    if stage not in CRM_STAGES:
        raise ValueError(f"stage must be one of {CRM_STAGES}")
    follow_up = datetime.now(timezone.utc) + timedelta(days=follow_up_days) if follow_up_days else None
    get_store().upsert_stage(customer_id, stage, notes, sentiment, follow_up)
    return True
