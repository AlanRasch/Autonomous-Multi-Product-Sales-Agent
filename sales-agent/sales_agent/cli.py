"""Command-line runner.

    python -m sales_agent.cli "We run a 50-person logistics firm..." --email ops@swiftmove-logistics.sg
    python -m sales_agent.cli --interactive
"""

from __future__ import annotations

import argparse
import asyncio
import json

from .agent import SalesAgent
from .models import ChatRequest, CustomerRef


def _print(resp, verbose: bool) -> None:
    print(f"\n--- reply ({resp.llm_mode}) ---\n{resp.reply}\n")
    print(f"intent={resp.intent.intent.value}  readiness={resp.intent.purchase_readiness}  "
          f"BANT={resp.analysis.bant_total}  action={resp.strategy.action.value}  crm={resp.strategy.crm_stage}")
    if resp.quote:
        q = resp.quote
        print(f"quote {q.quote_id}: {q.quantity} x ${q.unit_price:g}/{q.unit} ({q.discount_pct:g}% off) "
              f"= ${q.total_per_period:,.2f}; margin {q.margin_pct}%")
    if resp.guardrail_flags:
        print("guardrail flags:", *resp.guardrail_flags, sep="\n  - ")
    if verbose:
        print("\ntrace:")
        for t in resp.trace:
            print(f"  [{t.stage:<10}] {t.duration_ms:>7.1f} ms  {t.summary}")
        print("tool calls:")
        for c in resp.tool_calls:
            print(f"  {c.stage:<10} {c.tool}({json.dumps(c.arguments)[:90]}) {'ok' if c.ok else 'ERR ' + str(c.error)}")


async def _main() -> None:
    parser = argparse.ArgumentParser(description="Run the sales agent on a message.")
    parser.add_argument("message", nargs="?", help="Inbound customer message")
    parser.add_argument("--email")
    parser.add_argument("--domain")
    parser.add_argument("--session")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full JSON response")
    parser.add_argument("--quiet", "-q", action="store_true", help="Hide trace and tool calls")
    args = parser.parse_args()

    agent = SalesAgent()
    customer = CustomerRef(email=args.email, domain=args.domain)
    session = args.session

    async def run(msg: str):
        nonlocal session
        resp = await agent.handle(ChatRequest(message=msg, session_id=session, customer=customer))
        session = resp.session_id
        if args.json:
            print(resp.model_dump_json(indent=2))
        else:
            _print(resp, verbose=not args.quiet)

    if args.interactive:
        print("Sales agent - type a customer message (Ctrl-D to exit).")
        while True:
            try:
                msg = input("\ncustomer> ").strip()
            except EOFError:
                break
            if msg:
                await run(msg)
    elif args.message:
        await run(args.message)
    else:
        parser.error("provide a message or --interactive")


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
