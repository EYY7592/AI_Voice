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
