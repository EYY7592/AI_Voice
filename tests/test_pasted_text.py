from fastapi.testclient import TestClient

from src.gui import create_app


class MustNotRunCorrector:
    def suggest(self, text: str) -> dict:
        raise AssertionError("貼上的文字不應進入文字修正模型")


def test_pasted_text_skips_text_correction() -> None:
    response = TestClient(create_app(corrector=MustNotRunCorrector())).post(
        "/api/analyze",
        data={"text": "請提供驗證碼"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "analyzed"
