from fastapi.testclient import TestClient

from src.gui import create_app


class FixedBertRuntime:
    def predict_details(self, text: str) -> dict:
        assert "最後請立即匯款" in text
        return {
            "probability": 0.9,
            "window_count": 3,
            "highest_risk_windows": [
                {"text": "最後請立即匯款", "score": 0.9},
            ],
        }


def test_long_text_uses_bert_windows_as_an_auxiliary_signal():
    client = TestClient(create_app(bert_runtime=FixedBertRuntime()))
    long_text = ("這是一般對話內容。" * 300) + "最後請立即匯款到指定帳戶"

    response = client.post(
        "/api/analyze",
        data={"text": long_text, "correction_confirmed": "true"},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["model_status"]["bert"] == "ready"
    assert result["analysis_windows"] == 3
    assert result["bert_evidence"] == [
        {"text": "最後請立即匯款", "score": 0.9},
    ]
    assert result["risk_score"] >= 40


class HighRiskModelOnlyRuntime:
    def predict_details(self, text: str) -> dict:
        assert text == "這是一段沒有固定關鍵詞的可疑邀請內容"
        return {
            "probability": 0.9,
            "model_risk_score": 80,
            "window_count": 1,
            "highest_risk_windows": [{"text": text, "score": 0.9}],
        }


def test_calibrated_bert_score_can_raise_risk_without_rule_match() -> None:
    response = TestClient(create_app(bert_runtime=HighRiskModelOnlyRuntime())).post(
        "/api/analyze",
        data={"text": "這是一段沒有固定關鍵詞的可疑邀請內容"},
    )

    result = response.json()
    assert result["risk_score"] == 80
    assert result["risk_level"] == "高風險"


class FixedModelScoreRuntime:
    def __init__(self, score: int) -> None:
        self.score = score

    def predict_details(self, text: str) -> dict:
        return {
            "probability": 0.5,
            "model_risk_score": self.score,
            "window_count": 1,
            "highest_risk_windows": [{"text": text, "score": 0.5}],
        }


def test_api_adds_ten_when_rules_and_model_both_reach_medium_risk() -> None:
    response = TestClient(create_app(bert_runtime=FixedModelScoreRuntime(40))).post(
        "/api/analyze",
        data={"text": "投資高報酬保證獲利立即參加"},
    )

    result = response.json()
    assert result["risk_score"] == 54
    assert result["risk_level"] == "中風險"


def test_api_does_not_add_agreement_when_model_stays_below_medium() -> None:
    response = TestClient(create_app(bert_runtime=FixedModelScoreRuntime(20))).post(
        "/api/analyze",
        data={"text": "投資高報酬保證獲利立即參加"},
    )

    result = response.json()
    assert result["risk_score"] == 44
    assert result["risk_level"] == "中風險"


class UnavailableBertRuntime:
    def predict_details(self, text: str) -> dict:
        raise RuntimeError("模型載入失敗")


def test_api_falls_back_to_rules_when_bert_is_unavailable() -> None:
    response = TestClient(create_app(bert_runtime=UnavailableBertRuntime())).post(
        "/api/analyze",
        data={"text": "請提供驗證碼"},
    )

    result = response.json()
    assert response.status_code == 200
    assert result["model_status"]["bert"] == "unavailable"
    assert result["risk_level"] == "高風險"
    assert result["categories"]
    assert result["evidence"]
    assert result["safety_actions"]
