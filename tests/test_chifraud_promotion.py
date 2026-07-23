import json
import hashlib

import pytest
import torch


def _write_candidate(path, *, status="passed") -> None:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    model_dir = path / "model"
    model_dir.mkdir(parents=True)
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n詐\n騙\n", encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False).save_pretrained(model_dir)
    config = BertConfig(
        vocab_size=7,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        num_labels=2,
        id2label={0: "NORMAL", 1: "FRAUD"},
        label2id={"NORMAL": 0, "FRAUD": 1},
    )
    config.scamlens_calibration = {
        "temperature": 1.0,
        "medium_threshold": 0.4,
        "high_threshold": 0.8,
    }
    model = BertForSequenceClassification(config)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias[0] = -5.0
        model.classifier.bias[1] = 5.0
    model.save_pretrained(model_dir)
    artifact_files = []
    for artifact_path in sorted(item for item in model_dir.rglob("*") if item.is_file()):
        artifact_files.append(
            {
                "path": artifact_path.relative_to(model_dir).as_posix(),
                "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                "size": artifact_path.stat().st_size,
            }
        )
    passing_year = {"fraud_recall": 0.90, "medium_fpr": 0.12, "high_fpr": 0.05}
    manifest = {
        "status": status,
        "test_used_for_selection": False,
        "script_view": "traditional",
        "seed": 42,
        "artifact_files": artifact_files,
        "test_report": {
            "passed": status == "passed",
            "years": {"2022": passing_year, "2023": passing_year},
            "subtypes": {"1": {"samples": 50, "recall": 0.70}},
        },
    }
    (path / "training_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_validate_candidate_output_accepts_only_passed_calibrated_model(tmp_path) -> None:
    from src.chifraud_promotion import validate_candidate_output

    candidate = tmp_path / "candidate"
    _write_candidate(candidate)

    result = validate_candidate_output(candidate)

    assert result["status"] == "passed"
    assert result["fraud_label_id"] == 1


def test_validate_candidate_output_rejects_failed_acceptance(tmp_path) -> None:
    from src.chifraud_promotion import validate_candidate_output

    candidate = tmp_path / "candidate"
    _write_candidate(candidate, status="failed")

    with pytest.raises(ValueError, match="未通過離線驗收"):
        validate_candidate_output(candidate)


def test_validate_candidate_output_rejects_candidate_not_selected_by_report(tmp_path) -> None:
    from src.chifraud_promotion import validate_candidate_output

    candidate = tmp_path / "candidate"
    _write_candidate(candidate)
    selection_report = tmp_path / "selection_report.json"
    selection_report.write_text(
        json.dumps(
            {
                "selection": {"status": "selected", "candidate": "simplified", "promotion_seed": 42}
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="不是選模報告指定"):

        validate_candidate_output(candidate, selection_report_path=selection_report)

def test_validate_candidate_output_rejects_tampered_artifact(tmp_path) -> None:
    from src.chifraud_promotion import validate_candidate_output

    candidate = tmp_path / "candidate"
    _write_candidate(candidate)
    config_path = candidate / "model" / "config.json"
    config_path.write_text(config_path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(ValueError, match="hash 或大小不一致"):
        validate_candidate_output(candidate)
