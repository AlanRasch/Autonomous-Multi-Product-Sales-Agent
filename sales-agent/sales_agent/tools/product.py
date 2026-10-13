"""Product knowledge tools (mock catalog backed by data/products.json).

Swap the bodies for a vector store (Qdrant/Pinecone/Chroma) or SQL catalog in production.
"""

from __future__ import annotations

import re

from .. import data_loader
from ..models import PersonaPitch, Product, ProductSpec

_STOPWORDS = {
    "a", "an", "and", "are", "for", "in", "of", "on", "or", "the", "to", "we", "with",
    "looking", "need", "our", "us", "is", "can", "you", "real", "time",
}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _score(spec: ProductSpec, query: str) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    phrase_hits = sum(1 for uc in spec.use_cases if uc in query.lower())
    haystack = _tokens(
        " ".join([spec.name, spec.category, spec.summary, *spec.use_cases, *spec.target_industries])
    )
    overlap = len(q & haystack) / len(q)
    return round(min(1.0, overlap + 0.25 * phrase_hits), 3)


async def search_product_catalog(query: str, filters: dict | None = None) -> list[Product]:
    """Search the catalog by need/use case. Filters: market, category, industry, max_unit_price."""
    filters = filters or {}
    results: list[Product] = []
    for spec in data_loader.products().values():
        if filters.get("market") and spec.market not in (filters["market"], "B2B2C"):
            continue
        if filters.get("category") and filters["category"].lower() not in spec.category.lower():
            continue
        if filters.get("industry") and filters["industry"].lower() not in spec.target_industries:
            continue
        if filters.get("max_unit_price") is not None and spec.base_unit_price > filters["max_unit_price"]:
            continue
        score = _score(spec, query)
        if score > 0:
            results.append(Product(**spec.model_dump(include=set(Product.model_fields) - {"match_score"}), match_score=score))
    results.sort(key=lambda p: p.match_score or 0, reverse=True)
    return results[:5]


async def get_product_details(product_id: str) -> ProductSpec:
    spec = data_loader.products().get(product_id)
    if spec is None:
        raise ValueError(f"Unknown product_id '{product_id}'")
    return spec


async def get_value_propositions(product_id: str, target_persona: str) -> PersonaPitch:
    spec = await get_product_details(product_id)
    pitches = data_loader.intel()["persona_pitches"].get(product_id, {})
    persona_key = target_persona.lower().strip()
    pitch = next(
        (v for k, v in pitches.items() if k != "objections" and (k in persona_key or persona_key in k)),
        None,
    ) or next((v for k, v in pitches.items() if k != "objections"), None)
    objections = pitches.get("objections", {})
    if pitch is None:
        # Generic fallback derived from the spec - still grounded in catalog facts.
        return PersonaPitch(
            product_id=product_id,
            persona=target_persona,
            headline=spec.summary,
            outcomes=[],
            common_objections=objections,
            differentiators=spec.social_proof[:2],
        )
    return PersonaPitch(
        product_id=product_id,
        persona=target_persona,
        headline=pitch["headline"],
        outcomes=pitch["outcomes"],
        common_objections=objections,
        differentiators=pitch["differentiators"],
    )
