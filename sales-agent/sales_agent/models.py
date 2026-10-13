"""Pydantic models for every tool input/output, the agent state, and the HTTP API."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Product knowledge
# ---------------------------------------------------------------------------


class VolumeTier(BaseModel):
    min_quantity: int
    discount_pct: float


class PricingRules(BaseModel):
    base_unit_price: float
    currency: str = "USD"
    unit: str = Field(description="Billing unit, e.g. 'vehicle/month' or 'device'")
    unit_cost: float = Field(description="Internal cost per unit (used for margin checks)")
    volume_tiers: list[VolumeTier] = []
    annual_commit_discount_pct: float = 0.0
    max_discount_pct: float = Field(description="Hard ceiling on total authorized discount")
    min_margin_pct: float = Field(description="Margin floor a quote may never go under")


class Product(BaseModel):
    product_id: str
    name: str
    category: str
    market: Literal["B2B", "B2C", "B2B2C"]
    summary: str
    base_unit_price: float
    unit: str
    use_cases: list[str] = []
    target_industries: list[str] = []
    match_score: float | None = None


class ProductSpec(Product):
    features: list[str] = []
    pricing: PricingRules
    licensing_options: list[str] = []
    inventory_status: str = "available"
    regions: list[str] = []
    social_proof: list[str] = []


class PersonaPitch(BaseModel):
    product_id: str
    persona: str
    headline: str
    outcomes: list[str]
    common_objections: dict[str, str] = Field(description="objection -> recommended reframe")
    differentiators: list[str]


# ---------------------------------------------------------------------------
# Market & competitor intelligence
# ---------------------------------------------------------------------------


class WebResult(BaseModel):
    title: str
    snippet: str
    url: str | None = None
    published: str | None = None


class WebSearchResults(BaseModel):
    query: str
    geography: str
    results: list[WebResult]
    source: Literal["mock", "tavily"] = "mock"


class CompetitorMatrix(BaseModel):
    product_category: str
    competitor_name: str
    known: bool = True
    pricing_model: str | None = None
    list_price: float | None = None
    price_unit: str | None = None
    strengths: list[str] = []
    weaknesses: list[str] = []
    feature_gaps_vs_us: list[str] = []
    hidden_costs: list[str] = []


class IndustryReport(BaseModel):
    industry: str
    avg_annual_software_spend_usd: float | None = None
    common_tech_stack: list[str] = []
    pain_points: list[str] = []
    growth_rate_pct: float | None = None
    buying_committee: list[str] = []


# ---------------------------------------------------------------------------
# Customer & demographic intelligence
# ---------------------------------------------------------------------------


class DecisionMaker(BaseModel):
    name: str
    title: str
    role_in_deal: Literal["economic_buyer", "champion", "technical_buyer", "user", "unknown"] = "unknown"


class LeadProfile(BaseModel):
    domain: str
    company_name: str
    industry: str
    employee_count: int | None = None
    region: str | None = None
    tech_stack: list[str] = []
    funding_stage: str | None = None
    decision_makers: list[DecisionMaker] = []
    customer_id: str
    enriched: bool = Field(True, description="False when the profile is a synthetic placeholder")


class SegmentData(BaseModel):
    geography: str
    age_bracket: str
    income_tier: str
    purchasing_power_index: float
    spending_trends: list[str]
    cultural_preferences: list[str]


class Interaction(BaseModel):
    timestamp: datetime
    channel: str
    direction: Literal["inbound", "outbound"]
    summary: str
    sentiment: Literal["positive", "neutral", "negative"] = "neutral"


class ConversationThread(BaseModel):
    customer_id: str
    interactions: list[Interaction] = []
    purchased_products: list[str] = []
    open_tickets: int = 0
    current_stage: str | None = None


# ---------------------------------------------------------------------------
# Decision & execution
# ---------------------------------------------------------------------------


class Quote(BaseModel):
    quote_id: str
    product_id: str
    product_name: str
    quantity: int
    contract_term: Literal["monthly", "annual"]
    currency: str
    list_unit_price: float
    unit_price: float
    unit: str
    discount_pct: float
    discount_breakdown: dict[str, float]
    total_per_period: float
    margin_pct: float
    min_authorized_unit_price: float = Field(
        description="Lowest unit price the agent may ever state for this quantity/term"
    )
    discount_capped: bool = False
    rejected_discount_code: str | None = None
    valid_until: date
    notes: list[str] = []


class CalendarLink(BaseModel):
    customer_id: str
    duration_mins: int
    booking_url: str
    proposed_slots: list[datetime]


class CrmRecord(BaseModel):
    customer_id: str
    stage: str
    notes: list[str] = []
    sentiment: str | None = None
    follow_up_at: datetime | None = None
    updated_at: datetime


# ---------------------------------------------------------------------------
# Agent reasoning artifacts (one per state-machine stage)
# ---------------------------------------------------------------------------


class Intent(str, Enum):
    INBOUND_LEAD = "inbound_lead"
    COLD_OUTREACH_RESPONSE = "cold_outreach_response"
    NEGOTIATION = "negotiation"
    TECHNICAL_INQUIRY = "technical_inquiry"
    OBJECTION = "objection"


class Entities(BaseModel):
    company_name: str | None = None
    industry: str | None = None
    company_size: int | None = Field(None, description="Employee count if stated")
    location: str | None = None
    budget: str | None = None
    budget_amount: float | None = Field(None, description="Numeric budget or target unit price if stated")
    timeline: str | None = None
    quantity: int | None = Field(None, description="Units/seats/vehicles requested")
    desired_outcomes: list[str] = []
    pain_points: list[str] = []
    product_interest: str | None = Field(None, description="Short search query for the catalog")
    competitors_mentioned: list[str] = []
    competitor_price: float | None = None
    discount_code: str | None = None
    contract_term: Literal["monthly", "annual"] | None = None
    is_decision_maker: bool | None = None
    market: Literal["B2B", "B2C"] | None = None
    age_bracket: str | None = None
    income_tier: str | None = None


class IngestionResult(BaseModel):
    intent: Intent
    secondary_intents: list[Intent] = []
    purchase_readiness: Literal["research", "evaluating", "ready_to_buy"]
    objection_types: list[Literal["price", "complexity", "competitor", "timing", "trust", "other"]] = []
    entities: Entities
    summary: str


class EnrichmentBundle(BaseModel):
    products: list[Product] = []
    primary_product: ProductSpec | None = None
    persona_pitch: PersonaPitch | None = None
    lead: LeadProfile | None = None
    history: ConversationThread | None = None
    industry: IndustryReport | None = None
    competitors: list[CompetitorMatrix] = []
    market: WebSearchResults | None = None
    segment: SegmentData | None = None
    errors: list[str] = []


class BantScore(BaseModel):
    budget: int = Field(ge=0, le=25)
    authority: int = Field(ge=0, le=25)
    need: int = Field(ge=0, le=25)
    timeline: int = Field(ge=0, le=25)

    @property
    def total(self) -> int:
        return self.budget + self.authority + self.need + self.timeline


class ValueMapping(BaseModel):
    pain_point: str
    capability: str
    outcome: str


class AnalysisResult(BaseModel):
    bant: BantScore
    bant_total: int
    qualification: Literal["unqualified", "nurture", "qualified", "sales_ready"]
    fit_score: float = Field(ge=0, le=1)
    value_map: list[ValueMapping] = []
    quote: Quote | None = None
    margin_ok: bool = True
    can_beat_competitor: bool | None = None
    gaps: list[str] = Field(default=[], description="Unknowns to discover next")


class NextBestAction(str, Enum):
    INFORM_EDUCATE = "inform_educate"
    DISCOVERY = "discovery"
    HANDLE_OBJECTION = "handle_objection"
    PROPOSE_DEMO = "propose_demo"
    PROPOSE_QUOTE = "propose_quote"
    CLOSE = "close"


class StrategyDecision(BaseModel):
    action: NextBestAction
    angle: Literal["educational", "consultative", "competitive", "value_roi", "upsell", "close"]
    cta: str
    talking_points: list[str]
    crm_stage: str
    follow_up_days: int
    rationale: str


class ToolCallRecord(BaseModel):
    tool: str
    arguments: dict[str, Any]
    ok: bool
    duration_ms: float
    stage: str
    error: str | None = None


class StageTrace(BaseModel):
    stage: str
    duration_ms: float
    summary: str


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------


class CustomerRef(BaseModel):
    customer_id: str | None = None
    email: str | None = None
    domain: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20_000)
    session_id: str | None = None
    customer: CustomerRef = CustomerRef()
    channel: Literal["email", "chat", "web_form", "sms", "webhook"] = "chat"


class ChatResponse(BaseModel):
    session_id: str
    customer_id: str
    reply: str
    intent: IngestionResult
    analysis: AnalysisResult
    strategy: StrategyDecision
    quote: Quote | None = None
    meeting: CalendarLink | None = None
    crm: CrmRecord | None = None
    tool_calls: list[ToolCallRecord]
    guardrail_flags: list[str]
    trace: list[StageTrace]
    llm_mode: str
