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
        id2label = getattr(getattr(model, "config", None), "id2label", {})
        fraud_ids = [int(label_id) for label_id, label in id2label.items() if str(label).upper() == "FRAUD"]
        if len(fraud_ids) != 1:
            raise ValueError("BERT 模型必須明確且唯一地定義 FRAUD 標籤。")
        self.fraud_label_id = fraud_ids[0]
        label2id = getattr(model.config, "label2id", {})
        if label2id.get("FRAUD") != self.fraud_label_id:
            raise ValueError("BERT 模型的 id2label 與 label2id 標籤方向衝突。")
        calibration = getattr(model.config, "scamlens_calibration", None)
        if not isinstance(calibration, dict):
            raise ValueError("BERT 模型缺少 ScamLens 校準設定。")
        self.temperature = float(calibration.get("temperature", 0.0))
        self.medium_threshold = float(calibration.get("medium_threshold", 0.0))
        self.high_threshold = float(calibration.get("high_threshold", 0.0))
        if self.temperature <= 0 or not 0 < self.medium_threshold < self.high_threshold < 1:
            raise ValueError("BERT 模型的校準溫度或風險門檻無效。")

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
