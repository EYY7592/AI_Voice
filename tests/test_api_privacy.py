import tempfile

import numpy as np
from fastapi.testclient import TestClient

from src.gui import create_app
from src.media_extractors import AudioTextExtractor, MAX_IMAGE_BYTES
from src.models import TranscriptionResult


class FixedLoader:
    def load(self, path: str):
        with open(path, "rb") as handle:
            assert handle.read() == b"audio-bytes"
        return np.zeros(16_000, dtype=np.float32), 16_000


class PassthroughDenoiser:
    def denoise(self, audio, sample_rate):
        return audio


class FixedTranscriber:
    def transcribe(self, audio, sample_rate, language="zh"):
        return TranscriptionResult(text="客服要求提供驗證碼並匯款", confidence=0.9)


def test_audio_temp_file_is_removed_after_http_analysis(tmp_path, monkeypatch):
    model_path = tmp_path / "base.pt"
    model_path.write_bytes(b"model-placeholder")
    reader = AudioTextExtractor(
        model_path=model_path,
        loader=FixedLoader(),
        denoiser=PassthroughDenoiser(),
        transcriber=FixedTranscriber(),
    )
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    client = TestClient(create_app(audio_reader=reader))

    response = client.post(
        "/api/analyze",
        data={"correction_confirmed": "true"},
        files={"upload": ("call.wav", b"audio-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    assert sorted(path.name for path in tmp_path.iterdir()) == ["base.pt"]


def test_api_rejects_mixed_and_oversized_image_input():
    client = TestClient(create_app(image_reader=object()))

    mixed = client.post(
        "/api/analyze",
        data={"text": "這是一段足夠長的文字"},
        files={"upload": ("line.png", b"image", "image/png")},
    )
    oversized = client.post(
        "/api/analyze",
        files={"upload": ("line.png", b"0" * (MAX_IMAGE_BYTES + 1), "image/png")},
    )

    assert mixed.status_code == 422
    assert oversized.status_code == 413
