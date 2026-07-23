from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.bert_runtime import BertRuntime
from src.media_extractors import AudioTextExtractor, EasyOcrReader, MAX_AUDIO_SECONDS


class OneWindowTokenizer:
    def __call__(self, text, **kwargs):
        del text, kwargs
        return {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }

    def decode(self, ids, skip_special_tokens=True):
        del ids, skip_special_tokens
        return "測試窗口"


class DeclaredFraudLabelModel:
    config = SimpleNamespace(
        id2label={0: "NORMAL", 1: "FRAUD"},
        label2id={"NORMAL": 0, "FRAUD": 1},
        scamlens_calibration={
            "temperature": 1.0,
            "medium_threshold": 0.4,
            "high_threshold": 0.8,
        },
    )

    def __call__(self, **kwargs):
        del kwargs
        return type("Output", (), {"logits": torch.tensor([[5.0, 1.0]])})()


def test_bert_uses_declared_fraud_label() -> None:
    result = BertRuntime(OneWindowTokenizer(), DeclaredFraudLabelModel()).predict_details("測試")
    assert result["probability"] < 0.1


class ConflictingLabelModel:
    config = SimpleNamespace(
        id2label={0: "NORMAL", 1: "FRAUD"},
        label2id={"NORMAL": 1, "FRAUD": 0},
    )


def test_bert_rejects_conflicting_label_mapping() -> None:
    with pytest.raises(ValueError, match="衝突"):
        BertRuntime(OneWindowTokenizer(), ConflictingLabelModel())


class CalibratedHighRiskModel:
    config = SimpleNamespace(
        id2label={0: "NORMAL", 1: "FRAUD"},
        label2id={"NORMAL": 0, "FRAUD": 1},
        scamlens_calibration={
            "temperature": 1.0,
            "medium_threshold": 0.4,
            "high_threshold": 0.8,
        },
    )

    def __call__(self, **kwargs):
        del kwargs
        return type("Output", (), {"logits": torch.tensor([[0.0, 2.1972246]])})()


def test_bert_returns_calibrated_model_risk_score() -> None:
    result = BertRuntime(OneWindowTokenizer(), CalibratedHighRiskModel()).predict_details("測試")
    assert result["model_risk_score"] == 80


class TooLongLoader:
    def probe_duration(self, path: str) -> float:
        del path
        return MAX_AUDIO_SECONDS + 1

    def load(self, path: str, max_duration: float | None = None):
        raise AssertionError("超時音訊不得進入解碼")


def test_audio_duration_is_rejected_before_decode(tmp_path) -> None:
    reader = AudioTextExtractor(
        model_path=tmp_path / "base.pt",
        loader=TooLongLoader(),
        denoiser=object(),
        transcriber=object(),
    )
    with pytest.raises(ValueError, match="5 分鐘"):
        reader.extract(b"compressed-audio", ".mp3")


class HugeImage:
    size = (20_000, 20_000)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_ocr_rejects_excessive_decoded_pixels_before_model_load() -> None:
    reader = EasyOcrReader("unused")
    with patch("PIL.Image.open", return_value=HugeImage()):
        with pytest.raises(ValueError, match="像素"):
            reader.extract(b"compressed-image")
