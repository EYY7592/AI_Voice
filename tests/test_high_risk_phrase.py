from fastapi.testclient import TestClient

from src.gui import create_app


def test_requesting_a_verification_code_is_not_low_risk() -> None:
    response = TestClient(create_app()).post(
        "/api/analyze",
        data={"text": "請提供驗證碼"},
    )

    result = response.json()
    assert response.status_code == 200
    assert result["risk_level"] == "高風險"
    assert result["risk_score"] >= 70
    assert "釣魚／帳號驗證" in result["categories"]
