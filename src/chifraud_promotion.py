"""驗證並安全升級 localhost 使用的 ChiFraud BERT。"""
from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
import shutil


def _validate_model_dir(model_dir: Path) -> dict[str, object]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.bert_runtime import BertRuntime

    if not model_dir.is_dir():
        raise FileNotFoundError(f"找不到候選模型目錄：{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    runtime = BertRuntime(tokenizer, model, device="cpu")
    smoke = runtime.predict_details("這是一則本機候選模型載入測試。")
    return {
        "fraud_label_id": runtime.fraud_label_id,
        "calibration": dict(model.config.scamlens_calibration),
        "smoke_probability": smoke["probability"],
    }


def validate_candidate_output(
    candidate_dir: str | PathLike[str],
    *, selection_report_path: str | PathLike[str] | None = None,
) -> dict[str, object]:
    """拒絕未通過離線驗收或缺少校準契約的候選產物。"""
    candidate = Path(candidate_dir)
    manifest_path = candidate / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到訓練 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("候選模型未通過離線驗收，禁止升級 localhost。")
    if manifest.get("test_used_for_selection") is not False:
        raise ValueError("候選模型 manifest 未證明 test split 與選模隔離。")
    if selection_report_path is not None:
        report = json.loads(Path(selection_report_path).read_text(encoding="utf-8"))
        selection = report.get("selection", {})
        if selection.get("status") != "selected":
            raise ValueError("選模報告尚未選出可升級候選模型。")
        if (
            selection.get("candidate") != manifest.get("script_view")
            or int(selection.get("promotion_seed", -1)) != int(manifest.get("seed", -2))
        ):
            raise ValueError("此候選不是選模報告指定的字體與 promotion seed。")
    result = _validate_model_dir(candidate / "model")
    return {"status": "passed", "manifest": manifest, **result}


def promote_candidate(
    candidate_dir: str | PathLike[str],
    target_dir: str | PathLike[str],
    *, selection_report_path: str | PathLike[str],
) -> dict[str, str | None]:
    """以可回復的同層目錄切換模型；既有 previous 需先由人工處理。"""
    candidate = Path(candidate_dir).resolve()
    target = Path(target_dir).resolve()
    validate_candidate_output(candidate, selection_report_path=selection_report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f"{target.name}.next")
    previous = target.with_name(f"{target.name}.previous")
    if staging.exists():
        raise FileExistsError(f"暫存升級目錄已存在：{staging}")
    if target.exists() and previous.exists():
        raise FileExistsError(f"回復目錄已存在，請先確認後人工處理：{previous}")

    shutil.copytree(candidate / "model", staging)
    shutil.copy2(candidate / "training_manifest.json", staging / "training_manifest.json")
    try:
        _validate_model_dir(staging)
        if target.exists():
            target.rename(previous)
        staging.rename(target)
    except Exception:
        if not target.exists() and previous.exists():
            previous.rename(target)
        raise
    return {
        "target": str(target),
        "rollback": str(previous) if previous.exists() else None,
    }
