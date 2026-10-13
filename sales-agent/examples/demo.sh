#!/usr/bin/env bash
# Sends four representative events to a running server: uvicorn sales_agent.api:app
set -euo pipefail
URL="${URL:-http://localhost:8000/api/v1/agent/chat}"

send() {
  echo "=== $1"
  curl -s "$URL" -H 'content-type: application/json' -d "$2" |
    python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["reply"]); print("->", r["strategy"]["action"], "| stage:", r["strategy"]["crm_stage"], "| flags:", r["guardrail_flags"]); print()'
}

send "Negotiation vs competitor (spec few-shot)" \
  '{"message":"We run a 50-person logistics firm in Southeast Asia and are looking for real-time fleet tracking. CompetitorX quoted us $20/vehicle/month. Can you match it?","customer":{"email":"ops@swiftmove-logistics.sg"}}'

send "Ready to buy with a discount code" \
  '{"message":"We need a CRM with forecasting for 60 seats, budget approved, want to go live next month. Can you send a quote with code LAUNCH10?","customer":{"email":"dana@brightpath.io"}}'

send "Complexity objection" \
  '{"message":"Your backup product looks too complicated for our clinic, we are not sure.","customer":{"email":"priya@northside-clinic.org"}}'

send "B2C early research" \
  '{"message":"Just curious how solar panels could lower the energy bill for my home in Australia.","customer":{"email":"alex@gmail.com"}}'
