import os

import pytest

os.environ["SALES_AGENT_LLM_MODE"] = "mock"

from sales_agent.config import get_settings  # noqa: E402
from sales_agent.store import reset_store  # noqa: E402

FEW_SHOT = (
    "We run a 50-person logistics firm in Southeast Asia and are looking for real-time fleet "
    "tracking. CompetitorX quoted us $20/vehicle/month. Can you match it?"
)


@pytest.fixture(autouse=True)
def fresh_store():
    get_settings.cache_clear()
    return reset_store()
