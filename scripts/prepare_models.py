"""Explicitly download local OCR, correction, and Whisper weights."""
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def prepare_correction() -> None:
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    target = MODELS / "text_correction"
    target.mkdir(parents=True, exist_ok=True)
    name = "shibing624/macbert4csc-base-chinese"
    AutoTokenizer.from_pretrained(name).save_pretrained(target)
    AutoModelForMaskedLM.from_pretrained(name).save_pretrained(target)
    print(f"文字修正模型已準備：{target}")


def prepare_ocr() -> None:
    import easyocr

    target = MODELS / "ocr"
    target.mkdir(parents=True, exist_ok=True)
    easyocr.Reader(
        ["ch_tra", "en"],
        gpu=False,
        model_storage_directory=str(target),
        download_enabled=True,
    )
    print(f"EasyOCR 模型已準備：{target}")


def prepare_whisper() -> None:
    import whisper

    target = MODELS / "whisper"
    target.mkdir(parents=True, exist_ok=True)
    whisper.load_model("base", device="cpu", download_root=str(target))
    print(f"Whisper base 模型已準備：{target / 'base.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="準備 ScamLens-TW 本機模型")
    parser.add_argument(
        "component",
        nargs="?",
        choices=("all", "correction", "ocr", "whisper"),
        default="all",
    )
    component = parser.parse_args().component
    if component in ("all", "correction"):
        prepare_correction()
    if component in ("all", "ocr"):
        prepare_ocr()
    if component in ("all", "whisper"):
        prepare_whisper()


if __name__ == "__main__":
    main()
