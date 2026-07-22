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
            logits = self.model(**model_inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()

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
            "window_count": len(windows),
            "highest_risk_windows": highest,
        }

    def predict(self, text: str) -> tuple[float, int]:
        details = self.predict_details(text)
        return float(details["probability"]), int(details["window_count"])
