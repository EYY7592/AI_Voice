"""Opt-in local model smoke checks; never prints raw user content."""
from __future__ import annotations

import argparse
from pathlib import Path

from config.settings import settings
from src.bert_runtime import BertRuntime
from src.media_extractors import AudioTextExtractor, EasyOcrReader
from src.text_correction import TextCorrector


def main() -> int:
    parser = argparse.ArgumentParser(description="ScamLens-TW 本機模型 smoke check")
    parser.add_argument("--image", type=Path, help="清晰測試截圖")
    parser.add_argument("--audio", type=Path, help="五分鐘內測試語音")
    args = parser.parse_args()

    correction = TextCorrector(settings.models.correction).suggest("今天新情很好")
    print(f"[correction] status={correction['model_status']} changes={len(correction['changes'])}")

    bert = BertRuntime.load(settings.models.bert).predict_details("客服要求立即提供驗證碼")
    print(f"[bert] windows={bert['window_count']} score_available=True")

    if args.image:
        result = EasyOcrReader(settings.models.ocr).extract(args.image.read_bytes())
        print(f"[ocr] chars={len(str(result['text']))} confidence={result['confidence']}")
    if args.audio:
        result = AudioTextExtractor(model_path=settings.models.whisper).extract(
            args.audio.read_bytes(), args.audio.suffix.lower()
        )
        print(f"[whisper] chars={len(str(result['text']))} duration={result['duration']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
