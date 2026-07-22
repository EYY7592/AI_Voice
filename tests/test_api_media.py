from fastapi.testclient import TestClient

from src.gui import create_app


class FixedOcrReader:
    def extract(self, content: bytes) -> dict:
        assert content == b"image-bytes"
        return {"text": "客服要求提供驗證碼並立即匯款", "confidence": 0.91}


class FixedAudioReader:
    def extract(self, content: bytes, suffix: str) -> dict:
        assert content == b"audio-bytes"
        assert suffix == ".wav"
        return {"text": "投資老師保證獲利要求馬上入金", "duration": 12.5}


def test_image_and_audio_use_the_same_content_analysis_contract():
    client = TestClient(
        create_app(image_reader=FixedOcrReader(), audio_reader=FixedAudioReader())
    )

    image = client.post(
        "/api/analyze",
        data={"correction_confirmed": "true"},
        files={"upload": ("line.png", b"image-bytes", "image/png")},
    )
    audio = client.post(
        "/api/analyze",
        data={"correction_confirmed": "true"},
        files={"upload": ("call.wav", b"audio-bytes", "audio/wav")},
    )

    assert image.status_code == 200
    assert audio.status_code == 200
    for result, expected_type in ((image.json(), "image"), (audio.json(), "audio")):
        assert result["status"] == "analyzed"
        assert result["input_type"] == expected_type
        assert result["original_text"]
        assert result["risk_score"] >= 40
        assert result["evidence"]

    assert image.json()["extraction"]["confidence"] == 0.91
    assert audio.json()["extraction"]["duration"] == 12.5
