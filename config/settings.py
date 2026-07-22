"""ScamLens-TW localhost 設定。"""
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass(frozen=True)
class ModelConfig:
    bert: str = str(PROJECT_ROOT / "models" / "bert_fraud")
    correction: str = str(PROJECT_ROOT / "models" / "text_correction")
    ocr: str = str(PROJECT_ROOT / "models" / "ocr")
    whisper: str = str(PROJECT_ROOT / "models" / "whisper" / "base.pt")


@dataclass(frozen=True)
class AnalysisConfig:
    max_text_chars: int = 20_000
    max_image_bytes: int = 10 * 1024 * 1024
    max_audio_seconds: int = 300
    bert_window_tokens: int = 256
    bert_window_overlap: int = 64
    medium_risk_score: int = 40
    high_risk_score: int = 70


@dataclass(frozen=True)
class Settings:
    models: ModelConfig = field(default_factory=ModelConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    log_level: str = "INFO"


settings = Settings()
