"""驗證並安全升級 localhost 使用的 ChiFraud BERT。"""
from __future__ import annotations

import hashlib
import json
from os import PathLike
from pathlib import Path
import shutil


def _validate_model_dir(model_dir: Path) -> dict[str, object]:
    from fastapi.testclient import TestClient
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.bert_runtime import BertRuntime
    from src.gui import create_app

    if not model_dir.is_dir():
        raise FileNotFoundError(f"找不到候選模型目錄：{model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    runtime = BertRuntime(tokenizer, model, device="cpu")
    smoke = runtime.predict_details("這是一則本機候選模型載入測試。")
    long_smoke = runtime.predict_details("這是滑動窗口測試文字。" * 160)
    if int(long_smoke["window_count"]) < 2 or not long_smoke["highest_risk_windows"]:
        raise ValueError("候選模型未通過 256-token／64-token overlap 滑動窗口 smoke test。")
    if not 0.0 <= float(long_smoke["probability"]) <= 1.0 or not 0 <= int(long_smoke["model_risk_score"]) <= 90:
        raise ValueError("候選模型 smoke 分數超出契約範圍。")
    probe_details = max(
        (
            runtime.predict_details(text)
            for text in (
                "恭喜中獎，請先支付保證金領取獎金",
                "客服通知帳戶異常，請依指示提供資料",
                "穩賺投資方案，現在加入即可快速獲利",
            )
        ),
        key=lambda details: int(details["model_risk_score"]),
    )
    model_score = int(probe_details["model_risk_score"])
    if model_score < 70:
        raise ValueError("候選模型未能對 promotion 詐騙 probe 產生模型單獨高風險。")

    class FrozenCandidateRuntime:
        def predict_details(self, text: str) -> dict[str, object]:
            return probe_details

    client = TestClient(create_app(bert_runtime=FrozenCandidateRuntime()))
    model_only = client.post("/api/analyze", data={"text": "這是一段沒有固定關鍵詞的可疑邀請內容"})
    agreement = client.post("/api/analyze", data={"text": "投資高報酬保證獲利立即參加"})
    payload = model_only.json()
    required_fields = {"risk_score", "risk_level", "categories", "evidence", "safety_actions"}
    if (
        model_only.status_code != 200
        or payload.get("risk_score") != model_score
        or payload.get("risk_level") != "高風險"
        or not required_fields <= payload.keys()
    ):
        raise ValueError("候選模型未通過 localhost 模型單獨高風險 API smoke test。")
    expected_agreement = min(100, max(44, model_score) + 10)
    if agreement.status_code != 200 or agreement.json().get("risk_score") != expected_agreement:
        raise ValueError("候選模型未通過 localhost 規則與模型一致性加分 API smoke test。")
    return {
        "fraud_label_id": runtime.fraud_label_id,
        "calibration": dict(model.config.scamlens_calibration),
        "script_view": runtime.script_view,
        "smoke_probability": smoke["probability"],
        "window_count": long_smoke["window_count"],
    }


def _validate_artifact_files(candidate: Path, manifest: dict[str, object]) -> None:
    expected_files = manifest.get("artifact_files")
    if not isinstance(expected_files, list) or not expected_files:
        raise ValueError("候選模型 manifest 缺少 artifact 檔案 hash。")
    expected_paths: set[str] = set()
    for expected in expected_files:
        relative_path = str(expected.get("path", ""))
        artifact_path = candidate / "model" / relative_path
        if not relative_path or not artifact_path.is_file():
            raise ValueError(f"候選模型 artifact 缺檔：{relative_path}")
        expected_paths.add(relative_path)
        digest = hashlib.sha256()
        with artifact_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.get("sha256") or artifact_path.stat().st_size != int(expected.get("size", -1)):
            raise ValueError(f"候選模型 artifact hash 或大小不一致：{relative_path}")
    actual_paths = {
        path.relative_to(candidate / "model").as_posix()
        for path in (candidate / "model").rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("候選模型 artifact 實際檔案清單與 manifest 不一致。")


def _validate_test_report(manifest: dict[str, object]) -> None:
    report = manifest.get("test_report")
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError("候選模型 manifest 缺少通過的獨立 test 驗收報告。")
    years = report.get("years", {})
    for year in ("2022", "2023"):
        metrics = years.get(year, {})
        if (
            float(metrics.get("fraud_recall", -1)) < 0.90
            or float(metrics.get("medium_fpr", 2)) > 0.12
            or float(metrics.get("high_fpr", 2)) > 0.05
        ):
            raise ValueError(f"候選模型 {year} test 指標未通過 promotion 硬門檻。")
    for subtype, metrics in report.get("subtypes", {}).items():
        if int(metrics.get("samples", 0)) >= 50 and float(metrics.get("recall", -1)) < 0.70:
            raise ValueError(f"候選模型詐騙子類 {subtype} 未通過 promotion Recall 門檻。")


def validate_candidate_output(
    candidate_dir: str | PathLike[str],
    *,
    selection_report_path: str | PathLike[str] | None = None,
) -> dict[str, object]:
    """拒絕未通過離線驗收、檔案損壞或不符合選模結果的候選產物。"""
    candidate = Path(candidate_dir)
    manifest_path = candidate / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到訓練 manifest：{manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "passed":
        raise ValueError("候選模型未通過離線驗收，禁止升級 localhost。")
    if manifest.get("test_used_for_selection") is not False:
        raise ValueError("候選模型 manifest 未證明 test split 與選模隔離。")
    _validate_artifact_files(candidate, manifest)
    _validate_test_report(manifest)
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
    *,
    selection_report_path: str | PathLike[str],
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
    return {"target": str(target), "rollback": str(previous) if previous.exists() else None}
