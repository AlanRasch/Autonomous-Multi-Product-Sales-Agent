"""Local CRM + conversation store.

In-memory by default; set SALES_AGENT_CRM_STORE_PATH to persist to a JSON file.
Replace `CrmStore` with a HubSpot/Salesforce adapter for production use.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from .models import CrmRecord, Interaction


class CrmStore:
    def __init__(self, path: str = "") -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self.records: dict[str, CrmRecord] = {}
        self.interactions: dict[str, list[Interaction]] = {}
        self.sessions: dict[str, str] = {}  # session_id -> customer_id
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        self.records = {k: CrmRecord(**v) for k, v in raw.get("records", {}).items()}
        self.interactions = {
            k: [Interaction(**i) for i in v] for k, v in raw.get("interactions", {}).items()
        }
        self.sessions = raw.get("sessions", {})

    def _save(self) -> None:
        if not self._path:
            return
        payload = {
            "records": {k: v.model_dump(mode="json") for k, v in self.records.items()},
            "interactions": {
                k: [i.model_dump(mode="json") for i in v] for k, v in self.interactions.items()
            },
            "sessions": self.sessions,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- operations --------------------------------------------------------
    def bind_session(self, session_id: str, customer_id: str) -> None:
        with self._lock:
            self.sessions[session_id] = customer_id
            self._save()

    def customer_for_session(self, session_id: str) -> str | None:
        return self.sessions.get(session_id)

    def log_interaction(self, customer_id: str, interaction: Interaction) -> None:
        with self._lock:
            self.interactions.setdefault(customer_id, []).append(interaction)
            self._save()

    def upsert_stage(
        self,
        customer_id: str,
        stage: str,
        note: str,
        sentiment: str | None = None,
        follow_up_at: datetime | None = None,
    ) -> CrmRecord:
        with self._lock:
            now = datetime.now(timezone.utc)
            rec = self.records.get(customer_id)
            if rec is None:
                rec = CrmRecord(customer_id=customer_id, stage=stage, updated_at=now)
            rec.stage = stage
            if note:
                rec.notes.append(f"[{now:%Y-%m-%d %H:%M}] {note}")
            if sentiment:
                rec.sentiment = sentiment
            if follow_up_at:
                rec.follow_up_at = follow_up_at
            rec.updated_at = now
            self.records[customer_id] = rec
            self._save()
            return rec


_store: CrmStore | None = None


def get_store() -> CrmStore:
    global _store
    if _store is None:
        from .config import get_settings

        _store = CrmStore(get_settings().crm_store_path)
    return _store


def reset_store(path: str = "") -> CrmStore:
    """Replace the global store (used by tests)."""
    global _store
    _store = CrmStore(path)
    return _store
