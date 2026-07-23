"""本機 ChiFraud BERT 滑動窗口推論。"""
from __future__ import annotations

from pathlib import Path
from typing import Any


class BertRuntime:
    """以重疊窗口讀取長文字；最高風險窗口不被平均稀釋。"""

    def __init__(self, tokenizer: Any, model: Any, *, device: str = "cpu") -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        id2label = {int(label_id): str(label).upper() for label_id, label in getattr(model.config, "id2label", {}).items()}
        label2id = {str(label).upper(): int(label_id) for label, label_id in getattr(model.config, "label2id", {}).items()}
        if id2label != {0: "NORMAL", 1: "FRAUD"} or label2id != {"NORMAL": 0, "FRAUD": 1}:
            raise ValueError("BERT 模型標籤契約必須完整且唯一地定義 0=NORMAL、1=FRAUD，且雙向設定不得衝突。")
        self.fraud_label_id = 1
        calibration = getattr(model.config, "scamlens_calibration", None)
        if not isinstance(calibration, dict):
            raise ValueError("BERT 模型缺少 ScamLens 校準設定。")
        self.temperature = float(calibration.get("temperature", 0.0))
        self.medium_threshold = float(calibration.get("medium_threshold", 0.0))
        self.high_threshold = float(calibration.get("high_threshold", 0.0))
        if self.temperature <= 0 or not 0 < self.medium_threshold < self.high_threshold < 1:
            raise ValueError("BERT 模型的校準溫度或風險門檻無效。")
        self.script_view = str(getattr(model.config, "scamlens_script_view", "traditional"))
        if self.script_view not in {"simplified", "traditional"}:
            raise ValueError("BERT 模型的 scamlens_script_view 必須是 simplified 或 traditional。")
        self._script_converter = None
        if self.script_view == "simplified":
            import opencc
            self._script_converter = opencc.OpenCC("t2s")

    @classmethod
    def load(cls, model_path: str | Path, *, device: str | None = None) -> "BertRuntime":
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"找不到 BERT 模型：{path}")
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True)
        model.to(selected_device)
        model.eval()
        return cls(tokenizer, model, device=selected_device)

    def predict_details(self, text: str) -> dict[str, object]:
        import torch

        if self._script_converter is not None:
            text = self._script_converter.convert(text)
        encoded = self.tokenizer(
            text,
            max_length=256,
            stride=64,
            truncation=True,
            padding=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        encoded.pop("overflow_to_sample_mapping", None)
        input_ids = encoded["input_ids"]
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**model_inputs).logits / self.temperature
            probabilities = torch.softmax(logits, dim=-1)[:, self.fraud_label_id].detach().cpu().tolist()

        windows = [
            {
                "text": self.tokenizer.decode(ids, skip_special_tokens=True),
                "score": round(float(score), 4),
            }
            for ids, score in zip(input_ids, probabilities)
        ]
        highest = sorted(windows, key=lambda item: item["score"], reverse=True)[:3]
        return {
            "probability": max(probabilities, default=0.0),
            "model_risk_score": self._model_risk_score(max(probabilities, default=0.0)),
            "window_count": len(windows),
            "highest_risk_windows": highest,
        }

    def _model_risk_score(self, probability: float) -> int:
        probability = max(0.0, min(1.0, probability))
        anchors = (
            (0.0, 0),
            (self.medium_threshold, 40),
            (self.high_threshold, 70),
            (1.0, 90),
        )
        for (lower_p, lower_score), (upper_p, upper_score) in zip(anchors, anchors[1:]):
            if probability <= upper_p:
                return round(lower_score + (upper_score - lower_score) * (probability - lower_p) / (upper_p - lower_p))
        return 90

    def predict(self, text: str) -> tuple[float, int]:
        details = self.predict_details(text)
        return float(details["probability"]), int(details["window_count"])
