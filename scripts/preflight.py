"""在不下載任何內容的前提下檢查本機模型與部署版本。"""
import json
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
    bert_manifest = MODELS / "bert_fraud" / "training_manifest.json"
    for name, ready in CHECKS.items():
        print(f"[{'OK' if ready else 'MISSING'}] {name}")
    if bert_manifest.is_file():
        manifest = json.loads(bert_manifest.read_text(encoding="utf-8"))
        print(
            "[VERSION] ChiFraud BERT "
            f"base={manifest.get('base_revision')} "
            f"view={manifest.get('script_view')} seed={manifest.get('seed')}"
        )
    elif CHECKS["ChiFraud BERT"]:
        print("[VERSION] ChiFraud BERT 缺少 training_manifest.json，視為舊版模型。")
    if all(CHECKS.values()):
        return 0
    print("缺少模型時請執行：python scripts/prepare_models.py all")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
