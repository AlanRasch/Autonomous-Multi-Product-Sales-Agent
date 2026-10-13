"""Tool registry: typed argument models, Claude tool schemas and a safe executor."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..models import ToolCallRecord
from . import customer, execution, market, product


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CatalogFilters(_Args):
    market: Literal["B2B", "B2C"] | None = None
    category: str | None = None
    industry: str | None = None
    max_unit_price: float | None = None


class SearchProductCatalogArgs(_Args):
    query: str = Field(description="Need, use case or product description to search for")
    filters: CatalogFilters | None = None


class ProductIdArgs(_Args):
    product_id: str


class ValuePropArgs(_Args):
    product_id: str
    target_persona: str = Field(description="e.g. 'operations', 'executive', 'sales_leader', 'homeowner'")


class WebMarketArgs(_Args):
    query: str
    geography: str = "Global"


class CompetitorArgs(_Args):
    product_category: str
    competitor_name: str


class IndustryArgs(_Args):
    industry: str


class EnrichLeadArgs(_Args):
    email_or_domain: str


class SegmentArgs(_Args):
    geography: str
    age_bracket: str = "all"
    income_tier: str = "all"


class CustomerIdArgs(_Args):
    customer_id: str


class QuoteArgs(_Args):
    product_id: str
    quantity: int = Field(ge=1)
    discount_code: str | None = None
    contract_term: Literal["monthly", "annual"] = "annual"


class MeetingArgs(_Args):
    customer_id: str
    duration_mins: int = Field(30, ge=15, le=120)


class CrmArgs(_Args):
    customer_id: str
    stage: Literal[execution.CRM_STAGES]  # type: ignore[valid-type]
    notes: str = ""
    sentiment: Literal["positive", "neutral", "negative"] | None = None
    follow_up_days: int | None = Field(None, ge=1, le=90)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[..., Awaitable[Any]]
    category: Literal["product", "market", "customer", "execution"]

    def to_claude(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        return {"name": self.name, "description": self.description, "input_schema": schema}


TOOLS: dict[str, ToolSpec] = {
    t.name: t
    for t in [
        ToolSpec("search_product_catalog", "Search the internal product catalog for items matching a need, use case, spec or price point. Returns ranked products with base prices.", SearchProductCatalogArgs, product.search_product_catalog, "product"),
        ToolSpec("get_product_details", "Get full specs for one product: features, pricing tier rules, inventory status, regions, licensing and social proof.", ProductIdArgs, product.get_product_details, "product"),
        ToolSpec("get_value_propositions", "Get persona-specific messaging, common objections with recommended reframes, and differentiators for a product.", ValuePropArgs, product.get_value_propositions, "product"),
        ToolSpec("search_web_market_data", "Search current market trends, news, demand signals and regulatory changes for a topic and geography.", WebMarketArgs, market.search_web_market_data, "market"),
        ToolSpec("get_competitor_analysis", "Get a competitor's pricing model, list price, strengths, weaknesses, feature gaps and hidden costs.", CompetitorArgs, market.get_competitor_analysis, "market"),
        ToolSpec("get_industry_benchmarks", "Get industry benchmarks: average spend, common tech stack, pain points, growth rate and buying committee.", IndustryArgs, market.get_industry_benchmarks, "market"),
        ToolSpec("enrich_lead_profile", "Enrich a lead from an email or company domain: company size, industry, region, tech stack, funding and decision makers.", EnrichLeadArgs, customer.enrich_lead_profile, "customer"),
        ToolSpec("get_demographic_segment", "Get purchasing power, spending trends and cultural preferences for a geography / age / income segment.", SegmentArgs, customer.get_demographic_segment, "customer"),
        ToolSpec("get_interaction_history", "Get past interactions, purchases, open tickets and current CRM stage for a customer.", CustomerIdArgs, customer.get_interaction_history, "customer"),
        ToolSpec("calculate_dynamic_quote", "Calculate an authorized price quote using volume tiers, contract term, valid discount codes and margin floors. The ONLY source of truth for prices and discounts you may offer.", QuoteArgs, execution.calculate_dynamic_quote, "execution"),
        ToolSpec("schedule_meeting", "Find open calendar slots and return a booking link for a demo or call.", MeetingArgs, execution.schedule_meeting, "execution"),
        ToolSpec("update_crm_stage", "Update the customer's opportunity stage and add notes in the CRM.", CrmArgs, execution.update_crm_stage, "execution"),
    ]
}

# Tools exposed to the drafting LLM in the execution stage. CRM writes stay with the
# graph so every turn is logged exactly once, regardless of what the model does.
DRAFTING_TOOLS = [n for n in TOOLS if n != "update_crm_stage"]


def claude_tools(names: list[str] | None = None) -> list[dict[str, Any]]:
    return [TOOLS[n].to_claude() for n in (names or list(TOOLS))]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    return value


async def run_tool(name: str, arguments: dict[str, Any], stage: str) -> tuple[Any, ToolCallRecord]:
    """Validate arguments, run the tool and return (result, audit record). Never raises."""
    start = time.perf_counter()
    spec = TOOLS.get(name)
    try:
        if spec is None:
            raise ValueError(f"Unknown tool '{name}'")
        args = spec.args_model.model_validate(arguments)
        kwargs = {k: (v.model_dump(exclude_none=True) if isinstance(v, BaseModel) else v) for k, v in args}
        result = await spec.fn(**kwargs)
        ok, err = True, None
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        result, ok, err = None, False, str(exc)
    record = ToolCallRecord(
        tool=name,
        arguments=arguments,
        ok=ok,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        stage=stage,
        error=err,
    )
    return result, record


def result_to_text(result: Any, record: ToolCallRecord) -> str:
    if not record.ok:
        return f"Error: {record.error}"
    return json.dumps(_to_jsonable(result), default=str)
