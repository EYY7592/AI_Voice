import json
import pytest
import torch
from torch.nn import functional as F


from src.chifraud_training import (
    _as_cpu_rng_states,
    compute_sampling_weights,
    evaluate_candidate,
    evaluate_candidate_artifact,
    fit_temperature,
    recalibrate_candidate_artifact,
    train_candidate,
    select_script_candidate,
    select_operating_thresholds,
)


def test_cuda_rng_states_are_restored_as_cpu_byte_tensors() -> None:
    states = _as_cpu_rng_states([torch.tensor([1, 2], dtype=torch.uint8)])

    assert states[0].device.type == "cpu"
    assert states[0].dtype == torch.uint8


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
    assert manifest["dynamic_padding"] is True
    assert manifest["mixed_precision"] is False
    assert manifest["test_used_for_selection"] is False
    assert manifest["status"] in {"passed", "failed"}
    assert (tmp_path / "candidate" / "training_manifest.json").is_file()



def test_train_candidate_resumes_from_a_step_checkpoint(tmp_path) -> None:
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
    (prepared / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (prepared / "manifest.json").write_text(
        json.dumps({"source_revision": "fixture-v1"}),
        encoding="utf-8",
    )
    base_model = tmp_path / "tiny-bert"
    _write_tiny_bert(base_model)

    paused = train_candidate(
        prepared,
        tmp_path / "paused",
        base_model=base_model,
        max_epochs=1,
        batch_size=4,
        max_length=16,
        checkpoint_interval_steps=1,
        max_steps=1,
    )
    checkpoint = tmp_path / "paused" / "progress_checkpoint.pt"
    assert paused["status"] == "paused"
    assert paused["steps_completed"] == 1
    assert checkpoint.is_file()

    resumed = train_candidate(
        prepared,
        tmp_path / "resumed",
        base_model=base_model,
        max_epochs=1,
        batch_size=4,
        max_length=16,
        checkpoint_interval_steps=1,
        resume_checkpoint=checkpoint,
    )
    assert resumed["resumed_from_step"] == 1
    assert resumed["epochs_ran"] == 1
    assert resumed["status"] in {"passed", "failed"}


def test_equivalent_first_seed_requests_two_additional_seeds() -> None:
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

    assert decision["status"] == "additional_seeds_required"
    assert decision["additional_runs_per_candidate"] == 2


def test_equivalent_three_seed_average_selects_traditional_model() -> None:
    results = []
    for seed in (42, 43, 44):
        for candidate, offset in (("simplified", 0.004), ("traditional", 0.0)):
            for view in ("simplified", "traditional"):
                for year in (2022, 2023):
                    results.append(
                        {
                            "candidate": candidate,
                            "view": view,
                            "year": year,
                            "seed": seed,
                            "fraud_recall": 0.91 + offset,
                            "macro_f1": 0.90 + offset,
                        }
                    )

    decision = select_script_candidate(results)
    assert decision["candidate"] == "traditional"
    assert decision["promotion_seed"] == 42
    assert decision["comparison_summary"]


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


def test_candidate_artifact_recalibration_uses_validation_without_changing_weights(
    tmp_path, monkeypatch
) -> None:
    from hashlib import sha256
    from types import SimpleNamespace

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            values = [2 if "高" in text else 1 if "詐" in text else 0 for text in texts]
            return {"input_ids": torch.tensor(values).unsqueeze(1)}

    class FakeConfig:
        id2label = {0: "NORMAL", 1: "FRAUD"}
        label2id = {"NORMAL": 0, "FRAUD": 1}

        def save_pretrained(self, path):
            (path / "config.json").write_text(
                json.dumps({"scamlens_calibration": self.scamlens_calibration}),
                encoding="utf-8",
            )

    class FakeModel:
        def __init__(self):
            self.config = FakeConfig()

        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids):
            logits = torch.tensor([[3.0, 0.0], [0.0, 2.0], [0.0, 3.0]])
            return SimpleNamespace(logits=logits[input_ids.squeeze(1)])

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        lambda *args, **kwargs: FakeModel(),
    )

    model_dir = tmp_path / "candidate-model"
    model_dir.mkdir()
    weights = b"unchanged bert weights"
    (model_dir / "model.safetensors").write_bytes(weights)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    records = []
    for split in ("validation", "test"):
        for year in (2022, 2023):
            for text, label in (("正常", 0), ("詐騙", 1), ("高風險詐騙", 1)):
                records.append(
                    {
                        "year": year,
                        "binary_label": label,
                        "subtype_id": label,
                        "split": split,
                        "text_simplified": text,
                        "text_traditional": text,
                    }
                )
    (prepared / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    output = tmp_path / "recalibrated"
    manifest = recalibrate_candidate_artifact(
        model_dir,
        prepared,
        output,
        script_view="traditional",
        batch_size=3,
    )

    expected_hash = sha256(weights).hexdigest()
    assert manifest["test_used_for_calibration"] is False
    assert manifest["validation_samples"] == 6
    assert manifest["weights_sha256_before"] == expected_hash
    assert manifest["weights_sha256_after"] == expected_hash
    assert (output / "model.safetensors").read_bytes() == weights
