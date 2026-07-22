from fastapi.testclient import TestClient

from src.gui import create_app
from src.scam_analysis import ScamAnalyzer


def test_upload_rejects_extension_mime_mismatch() -> None:
    client = TestClient(create_app(image_reader=object()))
    response = client.post(
        "/api/analyze",
        files={"upload": ("message.png", b"not-an-image", "text/plain")},
    )
    assert response.status_code == 415


def test_risk_thresholds_are_configurable() -> None:
    analyzer = ScamAnalyzer(medium_risk_score=10, high_risk_score=20)
    result = analyzer.analyze_text("客服要求提供驗證碼")
    assert result.risk_level == "高風險"
