"""Post-generation compliance checks on the drafted reply.

- Every dollar figure must be traceable to a tool result or the customer's own message.
- No discount percentage above what `calculate_dynamic_quote` authorized.
- No unit price below the quote's `min_authorized_unit_price` (unless it is a cited competitor price).
- The reply must end with a call to action.
"""

from __future__ import annotations

import re

from .models import Quote

_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)")
_PERCENT_DISCOUNT = re.compile(r"(\d+(?:\.\d+)?)\s?%\s?(?:off|discount)", re.IGNORECASE)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_DATES = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ][\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?")


def _numbers(texts: list[str]) -> set[float]:
    found: set[float] = set()
    for text in texts:
        for raw in _NUMBER.findall(_DATES.sub(" ", text)):
            try:
                found.add(round(float(raw.replace(",", "")), 2))
            except ValueError:
                continue
    return found


def check_reply(reply: str, evidence: list[str], quote: Quote | None) -> list[str]:
    flags: list[str] = []
    allowed = _numbers(evidence)

    for raw in _MONEY.findall(reply):
        amount = round(float(raw.replace(",", "")), 2)
        if amount not in allowed:
            flags.append(f"unverified_figure: ${raw} does not appear in any tool result")

    if quote:
        # A price quoted per *our* unit below the authorized floor is a violation, unless it is
        # a figure taken from competitor intel (e.g. a competitor's add-on fee).
        competitor_numbers = _numbers([e for e in evidence if e.startswith("get_competitor_analysis")])
        unit_noun = re.escape(quote.unit.split("/")[0].rstrip("s"))
        per_unit = re.compile(rf"\$\s?(\d[\d,]*(?:\.\d+)?)\s*(?:per|/|a|an)\s*{unit_noun}", re.IGNORECASE)
        for raw in per_unit.findall(reply):
            amount = round(float(raw.replace(",", "")), 2)
            if amount < quote.min_authorized_unit_price and amount not in competitor_numbers:
                flags.append(
                    f"price_below_authorized: ${raw} is below the authorized ${quote.min_authorized_unit_price:g}"
                )
        for raw in _PERCENT_DISCOUNT.findall(reply):
            if float(raw) > quote.discount_pct + 1e-6:
                flags.append(f"discount_exceeds_authorized: {raw}% > authorized {quote.discount_pct:g}%")

    tail = reply.strip()[-400:]
    if "?" not in tail and "http" not in tail:
        flags.append("missing_cta: reply should end with a single clear call to action")
    return flags
