import json
import pytest
import torch
from torch.nn import functional as F


from src.chifraud_training import (
    compute_sampling_weights,
    evaluate_candidate,
    evaluate_candidate_artifact,
    fit_temperature,
    train_candidate,
    select_script_candidate,
    select_operating_thresholds,
)


def test_sampling_weights_balance_classes_and_mildly_boost_rare_subtypes() -> None:
    records = (
        [{"binary_label": 0, "subtype_id": 0}] * 200
        + [{"binary_label": 1, "subtype_id": 1}] * 100
        + [{"binary_label": 1, "subtype_id": 2}] * 25
    )

    weights = compute_sampling_weights(records)
    normal_mass = sum(weight for record, weight in zip(records, weights) if record["binary_label"] == 0)
    fraud_mass = sum(weight for record, weight in zip(records, weights) if record["binary_label"] == 1)
    common_weight = weights[200]
    rare_weight = weights[-1]

    assert normal_mass / fraud_mass == pytest.approx(2.0)
    assert rare_weight / common_weight == pytest.approx(2.0)
    assert rare_weight / common_weight <= 4.0


def test_thresholds_enforce_each_year_fpr_and_then_maximize_recall() -> None:
    records = []
    for year in (2022, 2023):
        for probability in [0.85] * 2 + [0.6] * 5 + [0.1] * 93:
            records.append({"year": year, "label": 0, "probability": probability})
        for probability in (0.55, 0.65, 0.9):
            records.append({"year": year, "label": 1, "probability": probability})

    thresholds = select_operating_thresholds(records)
    assert thresholds["medium_threshold"] == pytest.approx(0.55)
    assert thresholds["high_threshold"] == pytest.approx(0.65)


def test_candidate_acceptance_enforces_year_and_subtype_gates() -> None:
    records = []
    for year in (2022, 2023):
        for probability in [0.9] * 5 + [0.6] * 5 + [0.1] * 90:
            records.append({"year": year, "label": 0, "subtype_id": 0, "probability": probability})
        for probability in [0.9] * 90 + [0.1] * 10:
            records.append({"year": year, "label": 1, "subtype_id": 1, "probability": probability})

    report = evaluate_candidate(records, medium_threshold=0.5, high_threshold=0.8)

    assert report["passed"] is True
    assert report["years"]["2022"]["fraud_recall"] == pytest.approx(0.9)
    assert report["years"]["2023"]["medium_fpr"] == pytest.approx(0.1)
    assert report["years"]["2023"]["high_fpr"] == pytest.approx(0.05)
    assert report["subtypes"]["1"]["samples"] == 200
    assert report["subtypes"]["1"]["recall"] == pytest.approx(0.9)


def test_temperature_calibration_reduces_validation_nll() -> None:
    logits = torch.tensor([[5.0, 0.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    labels = torch.tensor([0, 1, 1, 0])

    temperature = fit_temperature(logits, labels)

    before = F.cross_entropy(logits, labels).item()
    after = F.cross_entropy(logits / temperature, labels).item()
    assert temperature > 1.0
    assert after < before


def _write_tiny_bert(path) -> None:
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    path.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "正", "常", "天", "氣", "詐", "騙", "匯", "款"]
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizer(vocab_file=str(vocab_path), do_lower_case=False)
    tokenizer.save_pretrained(path)
    model = BertForSequenceClassification(
        BertConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            num_labels=2,
            id2label={0: "NORMAL", 1: "FRAUD"},
            label2id={"NORMAL": 0, "FRAUD": 1},
        )
    )
    model.save_pretrained(path)


def test_train_candidate_runs_tiny_end_to_end_experiment(tmp_path) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    records = []
    for split in ("train", "validation", "test"):
        for year in (2022, 2023):
            for label, text in ((0, "正常天氣"), (1, "詐騙匯款")):
                for index in range(2):
                    records.append(
                        {
                            "record_id": f"{split}-{year}-{label}-{index}",
                            "year": year,
                            "subtype_id": label,
                            "subtype": "Normal" if label == 0 else "Gambling",
                            "binary_label": label,
                            "split": split,
                            "text_simplified": f"{text}{index}",
                            "text_traditional": f"{text}{index}",
                        }
                    )
    (prepared / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (prepared / "manifest.json").write_text(json.dumps({"source_revision": "fixture-v1"}), encoding="utf-8")
    base_model = tmp_path / "tiny-bert"
    _write_tiny_bert(base_model)

    manifest = train_candidate(prepared, tmp_path / "candidate", base_model=base_model, max_epochs=1, batch_size=4, max_length=16)
    assert manifest["epochs_ran"] == 1
    assert manifest["test_used_for_selection"] is False
    assert manifest["status"] in {"passed", "failed"}
    assert (tmp_path / "candidate" / "training_manifest.json").is_file()



def test_equivalent_script_candidates_select_traditional_model() -> None:
    results = []
    for candidate, offset in (("simplified", 0.004), ("traditional", 0.0)):
        for view in ("simplified", "traditional"):
            for year in (2022, 2023):
                results.append(
                    {
                        "candidate": candidate,
                        "view": view,
                        "year": year,
                        "seed": 42,
                        "fraud_recall": 0.91 + offset,
                        "macro_f1": 0.90 + offset,
                    }
                )

    decision = select_script_candidate(results)

    assert decision["status"] == "selected"
    assert decision["candidate"] == "traditional"
    assert decision["reason"] == "equivalent"


def test_candidate_artifact_can_be_evaluated_on_a_script_view(tmp_path) -> None:
    from transformers import AutoConfig

    model_dir = tmp_path / "model"
    _write_tiny_bert(model_dir)
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    config.scamlens_calibration = {
        "temperature": 1.0,
        "medium_threshold": 0.4,
        "high_threshold": 0.8,
    }
    config.save_pretrained(model_dir)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    records = []
    for year in (2022, 2023):
        for label, text in ((0, "正常天氣"), (1, "詐騙匯款")):
            records.append(
                {
                    "year": year,
                    "subtype_id": label,
                    "binary_label": label,
                    "split": "test",
                    "text_simplified": text,
                    "text_traditional": text,
                }
            )
    (prepared / "records.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    report = evaluate_candidate_artifact(model_dir, prepared, script_view="traditional", batch_size=2)
    assert set(report["years"]) == {"2022", "2023"}
