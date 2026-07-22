"""Check local model readiness without downloading anything."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
CHECKS = {
    "ChiFraud BERT": (MODELS / "bert_fraud" / "config.json").exists(),
    "文字修正模型": (MODELS / "text_correction" / "config.json").exists(),
    "EasyOCR": len(list((MODELS / "ocr").glob("*.pth"))) >= 2,
    "Whisper base": (MODELS / "whisper" / "base.pt").exists(),
}


def main() -> int:
    for name, ready in CHECKS.items():
        print(f"[{'OK' if ready else 'MISSING'}] {name}")
    if all(CHECKS.values()):
        return 0
    print("缺少模型時請執行：python scripts/prepare_models.py all")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
