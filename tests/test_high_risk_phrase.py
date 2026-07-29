import pytest
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


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("恭喜中獎，請先支付保證金領取獎金", "中獎／獎金詐騙"),
        ("客服通知帳戶異常，請依指示提供資料", "假客服／解除分期"),
        ("穩賺投資方案，現在加入即可快速獲利", "假投資"),
    ],
)
def test_obvious_scam_patterns_are_high_risk(text: str, category: str) -> None:
    result = TestClient(create_app()).post("/api/analyze", data={"text": text}).json()

    assert result["risk_level"] == "高風險"
    assert result["risk_score"] >= 70
    assert category in result["categories"]
