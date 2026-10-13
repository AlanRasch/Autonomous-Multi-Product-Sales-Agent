from conftest import FEW_SHOT
from fastapi.testclient import TestClient

from sales_agent.api import app


def test_chat_endpoint_round_trip():
    with TestClient(app) as client:
        assert client.get("/health").json()["llm_mode"] == "mock"

        resp = client.post("/api/v1/agent/chat", json={
            "message": FEW_SHOT, "customer": {"email": "ops@swiftmove-logistics.sg"},
        })
        assert resp.status_code == 200
        body = resp.json()
        assert body["quote"]["unit_price"] == 18.0
        assert body["strategy"]["action"] == "handle_objection"

        session = client.get(f"/api/v1/agent/sessions/{body['session_id']}").json()
        assert len(session["interactions"]) == 2
        crm = client.get(f"/api/v1/crm/{body['customer_id']}").json()
        assert crm["stage"] == "negotiation"

        assert client.post("/api/v1/agent/chat", json={"message": ""}).status_code == 422
        assert client.get("/api/v1/agent/sessions/nope").status_code == 404
