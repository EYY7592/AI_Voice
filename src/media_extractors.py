"""將本機圖片與語音轉成共同分析所需的文字。"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_AUDIO_SECONDS = 300.0


class EasyOcrReader:
    def __init__(self, model_dir: str | Path, *, gpu: bool | None = None) -> None:
        self.model_dir = Path(model_dir)
        self.gpu = gpu
        self._reader: Any = None

    def _load(self) -> Any:
        if self._reader is None:
            import torch
            import easyocr

            use_gpu = torch.cuda.is_available() if self.gpu is None else self.gpu
            self._reader = easyocr.Reader(
                ["ch_tra", "en"],
                gpu=use_gpu,
                model_storage_directory=str(self.model_dir),
                download_enabled=False,
            )
        return self._reader

    def extract(self, content: bytes) -> dict[str, object]:
        rows = self._load().readtext(content, detail=1, paragraph=False)
        texts = [str(row[1]).strip() for row in rows if len(row) >= 3 and str(row[1]).strip()]
        confidences = [float(row[2]) for row in rows if len(row) >= 3]
        return {
            "text": "\n".join(texts),
            "confidence": round(sum(confidences) / len(confidences), 4) if confidences else 0.0,
        }


class AudioTextExtractor:
    def __init__(
        self,
        *,
        model_path: str | Path,
        loader: Any | None = None,
        denoiser: Any | None = None,
        transcriber: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self._loader = loader
        self._denoiser = denoiser
        self._transcriber = transcriber

    def _load_services(self) -> None:
        if self._loader is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"找不到 Whisper base 模型：{self.model_path}")
        from src.step1_preprocessing.audio_loader import AudioLoader
        from src.step1_preprocessing.denoiser import Denoiser
        from src.step2_transcription.whisper_transcriber import WhisperTranscriber

        self._loader = AudioLoader(target_sr=16000)
        self._denoiser = Denoiser(prop_decrease=0.8)
        self._transcriber = WhisperTranscriber(model_size="base", model_path=str(self.model_path))

    def extract(self, content: bytes, suffix: str) -> dict[str, object]:
        self._load_services()
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(content)
                temp_path = Path(handle.name)
            audio, sample_rate = self._loader.load(str(temp_path))
            duration = len(audio) / sample_rate
            if duration > MAX_AUDIO_SECONDS:
                raise ValueError("語音不可超過 5 分鐘。")
            cleaned = self._denoiser.denoise(audio, sample_rate)
            transcript = self._transcriber.transcribe(cleaned, sample_rate, language="zh")
            return {
                "text": transcript.text,
                "duration": round(duration, 3),
                "confidence": transcript.confidence,
            }
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
