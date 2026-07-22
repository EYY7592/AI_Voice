"""完全使用本機權重的 Whisper 語音轉文字。"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import TranscriptionResult, TranscriptionSegment
from src.utils.exceptions import WhisperModelError
from src.utils.logger import get_logger


logger = get_logger("scamlens.whisper")


class WhisperTranscriber:
    MODEL_SIZES = {"tiny", "base", "small", "medium", "large"}

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        model_path: str | None = None,
    ) -> None:
        if model_size not in self.MODEL_SIZES:
            raise WhisperModelError(f"不支援的 Whisper 模型：{model_size}")
        self.model_size = model_size
        self.model_path = Path(model_path) if model_path else None
        self._model = None
        if device is None:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if self.model_path is None or not self.model_path.exists():
            raise WhisperModelError("Whisper base.pt 尚未準備，請先執行模型準備命令。")
        try:
            import whisper
            self._model = whisper.load_model(str(self.model_path), device=self.device)
        except Exception as exc:
            raise WhisperModelError(f"Whisper 本機模型載入失敗：{type(exc).__name__}") from exc

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "zh",
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        if sr != 16_000:
            raise WhisperModelError(f"Whisper 需要 16000 Hz，收到 {sr} Hz")
        self._load_model()
        try:
            result = self._model.transcribe(
                audio.astype(np.float32, copy=False),
                language=language,
                temperature=temperature,
                fp16=self.device == "cuda",
                verbose=False,
            )
        except Exception as exc:
            raise WhisperModelError(f"Whisper 轉錄失敗：{type(exc).__name__}") from exc

        segments: list[TranscriptionSegment] = []
        text_parts: list[str] = []
        total_logprob = 0.0
        total_tokens = 0
        speaker = "A"
        previous_end = 0.0
        for item in result.get("segments", []):
            start = float(item["start"])
            end = float(item["end"])
            if segments and start - previous_end > 0.8:
                speaker = "B" if speaker == "A" else "A"
            content = f"[角色 {speaker}]: {str(item['text']).strip()}"
            segments.append(TranscriptionSegment(start=start, end=end, text=content))
            text_parts.append(content)
            previous_end = end
            tokens = item.get("tokens", [])
            total_logprob += float(item.get("avg_logprob", -1.0)) * len(tokens)
            total_tokens += len(tokens)

        confidence = 0.0
        if total_tokens:
            confidence = min(1.0, max(0.0, 1.0 + (total_logprob / total_tokens) / 2.0))
        transcript = TranscriptionResult(
            text="\n".join(text_parts),
            segments=segments,
            language=str(result.get("language", language)),
            confidence=round(confidence, 4),
        )
        logger.info("語音轉錄完成：chars=%d, segments=%d", transcript.word_count, len(segments))
        return transcript

    def is_model_loaded(self) -> bool:
        return self._model is not None
