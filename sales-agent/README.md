# Autonomous Multi-Product Sales Agent

A goal-driven sales agent that adapts to any product catalog, industry or customer profile.
Each inbound message runs through a **5-stage LangGraph state machine**, calls a typed **tool
registry** (product, market, customer, execution), and drafts a reply with **Claude**, with
**compliance guardrails** on pricing and facts. It is exposed over **FastAPI** at
`POST /api/v1/agent/chat`.

```
User / Event ─► Context Orchestrator ─► [Market Intel | Product Knowledge | Customer CRM]
                                      ─► Decision & Pitch Engine ─► Action & Response Exec

[IDLE] ─► 1 INGESTION ─► 2 ENRICHMENT ─► 3 ANALYSIS ─► 4 STRATEGY ─► 5 EXECUTION ─► [WAIT/LISTEN]
```

It runs **with no API key**. Without credentials it uses a deterministic offline LLM: regex intent
extraction and template replies built only from tool results. You can demo, develop and test
everything locally, then set `ANTHROPIC_API_KEY` to switch to Claude.

---

## Quick start

```bash
cd sales-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # or: pip install -r requirements.txt

pytest                              # 15 tests, offline

# CLI (offline mock mode unless ANTHROPIC_API_KEY is set)
python -m sales_agent.cli "We run a 50-person logistics firm in Southeast Asia and are looking \
for real-time fleet tracking. CompetitorX quoted us \$20/vehicle/month. Can you match it?"

python -m sales_agent.cli -i --email mei@swiftmove-logistics.sg     # interactive multi-turn

# API
uvicorn sales_agent.api:app --reload
curl -s localhost:8000/api/v1/agent/chat -H 'content-type: application/json' \
  -d '{"message":"Can you match CompetitorX at $20/vehicle for 50 trucks?","customer":{"email":"ops@swiftmove-logistics.sg"}}' | jq .reply
```

Use Claude:

```bash
export ANTHROPIC_API_KEY=sk-ant-...       # or `ant auth login` + SALES_AGENT_LLM_MODE=claude
uvicorn sales_agent.api:app
```

Docker: `docker build -t sales-agent . && docker run -p 8000:8000 -e ANTHROPIC_API_KEY sales-agent`

---

## Project layout

```
sales_agent/
  api.py              FastAPI app: /api/v1/agent/chat, /webhooks/lead, /sessions/{id}, /crm/{id}, /health
  agent.py            SalesAgent.handle(ChatRequest) -> ChatResponse
  cli.py              One-shot and interactive CLI
  config.py           Settings (env vars, prefix SALES_AGENT_)
  models.py           Pydantic models for every tool I/O, stage artifact and API payload
  prompts.py          System prompt (verbatim from spec), few-shot example, stage instructions
  guardrails.py       Post-draft compliance checks
  store.py            Local CRM + conversation store (in-memory or JSON file)
  graph/
    state.py          AgentState (TypedDict + append-only reducers for audit trails)
    nodes.py          The 5 stages
    builder.py        LangGraph wiring
  llm/
    claude.py         Claude: structured-output intent extraction + tool-use drafting loop
    mock.py           Deterministic offline stand-in
  tools/
    product.py        search_product_catalog, get_product_details, get_value_propositions
    market.py         search_web_market_data (Tavily or mock), get_competitor_analysis, get_industry_benchmarks
    customer.py       enrich_lead_profile, get_demographic_segment, get_interaction_history
    execution.py      calculate_dynamic_quote, schedule_meeting, update_crm_stage
    registry.py       Typed arg models, Claude tool schemas, safe executor with audit records
  data/               Mock catalog (6 products, B2B and B2C), competitors, industries, leads, segments
tests/                Tools, guardrails, graph scenarios, API, Claude loop (stubbed client)
```

---

## The state machine

| Stage | What happens | Tools |
|---|---|---|
| **1 Ingestion** | Resolves the customer and session, loads history, and classifies intent (`inbound_lead`, `cold_outreach_response`, `negotiation`, `technical_inquiry`, `objection`) and purchase readiness. Extracts entities: budget, timeline, quantity, location, pain points, competitors and competitor price, discount code, contract term. On Claude this is a structured output validated against `IngestionResult`. | – |
| **2 Enrichment** | Two **parallel** waves. Wave 1: catalog search, lead enrichment, interaction history, demographic segment. Wave 2, which depends on the top product and industry: product details, persona value props, competitor matrices, industry benchmarks, market news. | 9 read tools |
| **3 Analysis** | Scores **BANT** (0–100) into `unqualified`, `nurture`, `qualified` or `sales_ready`. Computes the product fit score, maps pain points to capabilities to outcomes, calls **`calculate_dynamic_quote`** with margin checks, compares against the competitor price, and lists the discovery gaps. | `calculate_dynamic_quote` |
| **4 Strategy** | Picks the next best action: `discovery`, `inform_educate`, `handle_objection`, `propose_demo`, `propose_quote` or `close`. Also sets the angle (educational, consultative, competitive, value/ROI, upsell, close), a single CTA, the CRM stage and the follow-up cadence. These are deterministic, auditable rules. | – |
| **5 Execution** | Books meeting slots when the CTA calls for one, then drafts the reply. On Claude this is a tool-use loop where the model may call any read or quote tool to verify facts. Runs the guardrails and, on Claude, does one automatic revision if any check fails. Writes the CRM stage, notes, sentiment and follow-up, and logs both sides of the conversation. | `schedule_meeting`, `update_crm_stage` (+ any tool the model calls) |

Every response includes the `trace` (per-stage timing and summary), every `tool_calls` record
(arguments, success, latency, stage) and any `guardrail_flags`, so each decision can be audited.

### Guardrails

The system prompt's compliance rules are also enforced in code, not left to the prompt alone:

- **Discount authority:** `calculate_dynamic_quote` is the only source of prices. Discounts stack in
  this order: volume tier, then annual commitment, then a valid code. The total is capped at
  `min(max_discount_pct, margin-floor discount)`. Anything above the cap is noted as needing
  sales-manager approval.
- **No below-floor prices:** any `$X per <unit>` in the reply below the quote's
  `min_authorized_unit_price` is flagged, unless the figure comes from competitor intel.
- **Never guess figures:** every `$` amount in the reply must appear in a tool result or in the
  customer's own message.
- **Discount %:** any "N% off" above the authorized percentage is flagged.
- **Call to action:** the reply must end with one.

On Claude, flags trigger one revision pass with the violations as feedback. Any flags that remain
are returned in `guardrail_flags`.

---

## API

`POST /api/v1/agent/chat`

```json
{
  "message": "We run a 50-person logistics firm in Southeast Asia ... Can you match it?",
  "session_id": null,
  "customer": {"email": "ops@swiftmove-logistics.sg", "domain": null, "customer_id": null},
  "channel": "chat"
}
```

The response is abridged here:

```json
{
  "session_id": "4f1c...",
  "customer_id": "cust-swiftmove",
  "reply": "Hi Mei, ... we can offer $18 per vehicle/month, below CompetitorX's quote ... Would you be open to a 15-minute live preview with a small pilot on Thursday? ...",
  "intent":   {"intent": "negotiation", "purchase_readiness": "ready_to_buy", "objection_types": ["price","competitor"], "entities": {...}},
  "analysis": {"bant_total": 80, "qualification": "sales_ready", "fit_score": 0.64, "value_map": [...], "quote": {...}, "can_beat_competitor": true},
  "strategy": {"action": "handle_objection", "angle": "competitive", "cta": "...", "crm_stage": "negotiation"},
  "quote":    {"unit_price": 18.0, "discount_pct": 28.0, "total_per_period": 900.0, "margin_pct": 38.9, ...},
  "meeting":  {"booking_url": "https://cal.example.com/book/cust-swiftmove?duration=15", "proposed_slots": [...]},
  "crm":      {"stage": "negotiation", "notes": [...], "follow_up_at": "..."},
  "tool_calls": [...], "guardrail_flags": [], "trace": [...], "llm_mode": "claude"
}
```

Other endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/webhooks/lead` | Same pipeline, for form fills and inbound email events |
| `GET /api/v1/agent/sessions/{session_id}` | Conversation history for a session |
| `GET /api/v1/crm/{customer_id}` | The customer's CRM record |
| `GET /health` | Health check |

Interactive docs are served at `/docs`.

---

## Configuration

Set these as environment variables or in a `.env` file. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | – | Present: Claude mode. Absent: offline mock |
| `SALES_AGENT_LLM_MODE` | `auto` | `auto`, `claude` or `mock` |
| `SALES_AGENT_MODEL` | `claude-opus-5-5` | Any current Claude model ID |
| `SALES_AGENT_EFFORT` | `medium` | Drafting effort (`low` to `max`). Ingestion always uses `low` |
| `SALES_AGENT_ENABLE_FALLBACKS` | `true` | Server-side refusal fallback (`fallbacks="default"`). Set `false` on Bedrock, Vertex or Foundry |
| `SALES_AGENT_MAX_TOOL_ITERATIONS` | `8` | Cap on the drafting tool loop |
| `SALES_AGENT_CRM_STORE_PATH` | – | JSON file to persist the CRM and conversations |
| `SALES_AGENT_BOOKING_BASE_URL` | `https://cal.example.com/book` | Your scheduling link base |
| `SALES_AGENT_TAVILY_API_KEY` | – | Enables live web market search |

---

## Making it production-ready

Each mock is isolated behind the tool function signature, so replacements don't touch the graph:

| Mock | Replace with |
|---|---|
| `data/products.json` + keyword search | Vector store (Qdrant, Pinecone or Chroma) over your catalog, or a SQL query |
| `enrich_lead_profile` | Clearbit, Apollo or ZoomInfo API |
| `CrmStore` / `update_crm_stage` | HubSpot or Salesforce client. Keep the stage list in `tools/execution.py::CRM_STAGES` |
| `schedule_meeting` | Calendly, Cal.com or Google Calendar free/busy |
| `search_web_market_data` | Already supports Tavily; add others in `tools/market.py` |
| `calculate_dynamic_quote` | Your CPQ or pricing service. Keep returning `min_authorized_unit_price` |

To add a product, append it to `products.json`. Optionally add persona pitches and objection
reframes under `persona_pitches` in `intel.json`. No code changes are needed.

To add a tool, write an async function, an `_Args` model and a `ToolSpec` entry in
`tools/registry.py`. Claude sees it automatically in the drafting loop.
