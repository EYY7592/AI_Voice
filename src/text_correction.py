"""本機文字修正候選產生器；任何修改都必須由使用者確認。"""
from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import opencc

from src.step2_transcription.text_converter import TextConverter


class TextCorrector:
    def __init__(self, model_path: str | Path, *, device: str | None = None) -> None:
        self.model_path = Path(model_path)
        self.device = device
        self._tokenizer: Any = None
        self._model: Any = None
        self._normalizer = TextConverter("s2twp")
        self._to_simplified = opencc.OpenCC("t2s")

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if not self.model_path.exists():
            return False
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModelForMaskedLM.from_pretrained(self.model_path, local_files_only=True)
        self._model.to(self.device)
        self._model.eval()
        return True

    def suggest(self, text: str) -> dict[str, object]:
        normalized = self._normalizer.process(text)
        status = "unavailable"
        suggested = normalized
        try:
            if self._load():
                suggested = self._correct_chunks(normalized)
                status = "ready"
        except Exception:
            status = "unavailable"
            suggested = normalized
        return {
            "original_text": text,
            "suggested_text": suggested,
            "changes": self._changes(text, suggested),
            "model_status": status,
        }

    def _correct_chunks(self, text: str) -> str:
        return "".join(self._correct_chunk(text[start:start + 200]) for start in range(0, len(text), 200))

    def _correct_chunk(self, text: str) -> str:
        import torch

        simplified = self._to_simplified.convert(text)
        encoded = self._tokenizer(simplified, return_tensors="pt", truncation=True, max_length=256)
        model_inputs = {name: value.to(self.device) for name, value in encoded.items()}
        with torch.no_grad():
            predicted = self._model(**model_inputs).logits.argmax(dim=-1)[0]
        corrected = self._tokenizer.decode(predicted, skip_special_tokens=True).replace(" ", "")
        if len(corrected) != len(simplified):
            return self._normalizer.process(text)
        return self._normalizer.process(corrected)

    @staticmethod
    def _changes(original: str, suggested: str) -> list[dict[str, object]]:
        changes: list[dict[str, object]] = []
        for operation, i1, i2, j1, j2 in SequenceMatcher(None, original, suggested).get_opcodes():
            if operation == "equal":
                continue
            changes.append({
                "before": original[i1:i2],
                "after": suggested[j1:j2],
                "start": i1,
                "end": i2,
            })
        return changes
