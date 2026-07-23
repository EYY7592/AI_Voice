from fastapi.testclient import TestClient

from src.gui import create_app


class FixedCorrector:
    def suggest(self, text: str) -> dict:
        return {
            "original_text": text,
            "suggested_text": text.replace("新情", "心情"),
            "changes": [{"before": "新情", "after": "心情"}],
            "model_status": "ready",
        }

class FixedOcrReader:
    def extract(self, content: bytes) -> dict:
        assert content == b"image-bytes"
        return {"text": "今天新情很好但對方要求匯款", "confidence": 0.9}


def test_correction_is_proposed_before_it_can_change_analysis_text():
    client = TestClient(create_app(corrector=FixedCorrector(), image_reader=FixedOcrReader()))

    proposed = client.post(
        "/api/analyze",
        files={"upload": ("message.png", b"image-bytes", "image/png")},
    )

    assert proposed.status_code == 200
    proposal = proposed.json()
    assert proposal["status"] == "needs_confirmation"
    assert proposal["original_text"] == "今天新情很好但對方要求匯款"
    assert proposal["suggested_text"] == "今天心情很好但對方要求匯款"
    assert proposal["corrections"] == [{"before": "新情", "after": "心情"}]
    assert proposal["risk_score"] is None

    confirmed = client.post(
        "/api/analyze",
        data={
            "text": proposal["suggested_text"],
            "correction_confirmed": "true",
        },
    )

    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "analyzed"
    assert confirmed.json()["analysis_text"] == "今天心情很好但對方要求匯款"
