"""Market & competitor intelligence tools.

`search_web_market_data` uses the Tavily API when SALES_AGENT_TAVILY_API_KEY is set,
otherwise curated mock news from data/intel.json.
"""

from __future__ import annotations

import logging

import httpx

from .. import data_loader
from ..config import get_settings
from ..models import CompetitorMatrix, IndustryReport, WebResult, WebSearchResults

log = logging.getLogger(__name__)

_INDUSTRY_ALIASES = {
    "transport": "logistics", "transportation": "logistics", "shipping": "logistics",
    "trucking": "logistics", "delivery": "logistics", "software": "saas", "tech": "saas",
    "hospital": "healthcare", "clinic": "healthcare", "medical": "healthcare",
    "ecommerce": "retail", "e-commerce": "retail", "factory": "manufacturing",
    "household": "consumer", "home": "consumer", "residential": "consumer",
}


def normalize_industry(industry: str | None) -> str:
    if not industry:
        return "unknown"
    key = industry.lower().strip()
    known = data_loader.intel()["industries"]
    if key in known:
        return key
    for alias, target in _INDUSTRY_ALIASES.items():
        if alias in key:
            return target
    for name in known:
        if name in key:
            return name
    return key


async def _tavily(query: str, geography: str, api_key: str) -> WebSearchResults:
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={"api_key": api_key, "query": f"{query} {geography}".strip(), "max_results": 5},
        )
        resp.raise_for_status()
        data = resp.json()
    return WebSearchResults(
        query=query,
        geography=geography,
        source="tavily",
        results=[
            WebResult(title=r.get("title", ""), snippet=r.get("content", "")[:400], url=r.get("url"))
            for r in data.get("results", [])
        ],
    )


async def search_web_market_data(query: str, geography: str = "Global") -> WebSearchResults:
    settings = get_settings()
    if settings.tavily_api_key:
        try:
            return await _tavily(query, geography, settings.tavily_api_key)
        except httpx.HTTPError as exc:  # fall back to mock data rather than failing the turn
            log.warning("Tavily search failed, using mock market data: %s", exc)

    q = query.lower()
    geo = (geography or "Global").lower()
    hits = []
    for item in data_loader.intel()["market_news"]:
        kw_hits = sum(1 for kw in item["keywords"] if kw in q)
        if not kw_hits:
            continue
        geo_bonus = 1 if item["geography"].lower() in (geo, "global") or geo in item["geography"].lower() else 0
        hits.append((kw_hits + geo_bonus, item))
    hits.sort(key=lambda h: h[0], reverse=True)
    return WebSearchResults(
        query=query,
        geography=geography,
        results=[
            WebResult(title=i["title"], snippet=i["snippet"], published=i["published"]) for _, i in hits[:4]
        ],
    )


async def get_competitor_analysis(product_category: str, competitor_name: str) -> CompetitorMatrix:
    name = competitor_name.lower().replace(" ", "")
    for comp in data_loader.intel()["competitors"]:
        if comp["competitor_name"].lower().replace(" ", "") == name:
            return CompetitorMatrix(**comp)
    return CompetitorMatrix(
        product_category=product_category,
        competitor_name=competitor_name,
        known=False,
        weaknesses=[],
        strengths=[],
    )


async def get_industry_benchmarks(industry: str) -> IndustryReport:
    key = normalize_industry(industry)
    data = data_loader.intel()["industries"].get(key)
    if data is None:
        return IndustryReport(industry=key)
    return IndustryReport(industry=key, **data)
