"""Loads the bundled mock datasets. Swap these for real catalog/CRM/market sources."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import ProductSpec

DATA_DIR = Path(__file__).parent / "data"


def _load(name: str) -> Any:
    return json.loads((DATA_DIR / name).read_text(encoding="utf-8"))


@lru_cache
def products() -> dict[str, ProductSpec]:
    return {p["product_id"]: ProductSpec(**p) for p in _load("products.json")}


@lru_cache
def intel() -> dict[str, Any]:
    return _load("intel.json")


@lru_cache
def leads() -> dict[str, dict[str, Any]]:
    return {lead["domain"]: lead for lead in _load("leads.json")}
