from fastapi.testclient import TestClient

from src.gui import create_app


def test_user_can_analyze_pasted_text_without_binary_probability():
    client = TestClient(create_app())

    response = client.post(
        "/api/analyze",
        data={
            "text": "客服要求我立即提供驗證碼並匯款到指定帳戶",
            "correction_confirmed": "true",
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "analyzed"
    assert result["input_type"] == "text"
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in {"低風險", "中風險", "高風險"}
    assert result["categories"]
    assert result["evidence"]
    assert result["safety_actions"]
    assert "fraud_probability" not in result
    assert result["disclaimer"]
