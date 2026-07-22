"""語音轉錄資料契約。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TranscriptionSegment:
    start: float = 0.0
    end: float = 0.0
    text: str = ""


@dataclass
class TranscriptionResult:
    text: str = ""
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "zh"
    confidence: float = 0.0

    @property
    def word_count(self) -> int:
        return len(self.text.replace(" ", ""))

    @property
    def total_duration(self) -> float:
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start
