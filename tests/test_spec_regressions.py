from fastapi.testclient import TestClient

from src.gui import create_app


def test_insufficient_data_still_returns_safe_next_actions() -> None:
    response = TestClient(create_app()).post(
        "/api/analyze",
        data={"text": "你好", "correction_confirmed": "true"},
    )
    result = response.json()
    assert result["status"] == "insufficient_data"
    assert result["risk_score"] is None
    assert result["safety_actions"]
