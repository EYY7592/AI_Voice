"""ChiFraud BERT 候選模型的訓練與驗收。"""
from __future__ import annotations

from collections import Counter
import hashlib
import math
from os import PathLike
import platform
from typing import Mapping, Sequence
from statistics import fmean, pstdev


def _as_cpu_rng_states(states):
    return [state.cpu() for state in states]

def compute_sampling_weights(
    records: Sequence[Mapping[str, object]], *, subtype_weight_cap: float = 4.0
) -> list[float]:
    """讓每輪正常／詐騙總權重為 2:1，並溫和補償低頻詐騙子類。"""
    normal_count = sum(int(record["binary_label"]) == 0 for record in records)
    subtype_counts = Counter(
        int(record["subtype_id"])
        for record in records
        if int(record["binary_label"]) == 1
    )
    if normal_count == 0 or not subtype_counts:
        raise ValueError("取樣資料必須同時包含正常與詐騙樣本。")
    if subtype_weight_cap < 1:
        raise ValueError("詐騙子類補償上限不得小於 1。")
    largest_subtype = max(subtype_counts.values())
    fraud_weights = {
        subtype_id: min(subtype_weight_cap, math.sqrt(largest_subtype / count))
        for subtype_id, count in subtype_counts.items()
    }
    fraud_mass = sum(fraud_weights[subtype_id] * count for subtype_id, count in subtype_counts.items())
    return [
        2.0 / normal_count
        if int(record["binary_label"]) == 0
        else fraud_weights[int(record["subtype_id"])] / fraud_mass
        for record in records
    ]


def select_operating_thresholds(
    records: Sequence[Mapping[str, object]],
    *,
    medium_fpr_limit: float = 0.10,
    high_fpr_limit: float = 0.03,
) -> dict[str, float]:
    """在每年度正常誤報護欄內，選擇詐騙 Recall 最高的中／高門檻。"""
    if not 0 < high_fpr_limit < medium_fpr_limit < 1:
        raise ValueError("門檻校準的 FPR 限制必須滿足 0 < high < medium < 1。")
    normalized = [
        {
            "year": int(record["year"]),
            "label": int(record["label"]),
            "probability": float(record["probability"]),
        }
        for record in records
    ]
    if {record["year"] for record in normalized} != {2022, 2023}:
        raise ValueError("門檻校準必須同時包含 2022 與 2023 validation。")
    for record in normalized:
        if record["label"] not in (0, 1) or not 0 <= record["probability"] <= 1:
            raise ValueError("validation 標籤或機率超出有效範圍。")
    for year in (2022, 2023):
        if {record["label"] for record in normalized if record["year"] == year} != {0, 1}:
            raise ValueError(f"{year} validation 必須同時包含正常與詐騙樣本。")

    candidates = sorted({record["probability"] for record in normalized})

    def recall(threshold: float) -> float:
        fraud = [record for record in normalized if record["label"] == 1]
        return sum(record["probability"] >= threshold for record in fraud) / len(fraud)

    def within_fpr(threshold: float, limit: float) -> bool:
        for year in (2022, 2023):
            normal = [record for record in normalized if record["year"] == year and record["label"] == 0]
            if sum(record["probability"] >= threshold for record in normal) / len(normal) > limit:
                return False
        return True

    medium_candidates = [threshold for threshold in candidates if within_fpr(threshold, medium_fpr_limit)]
    if not medium_candidates:
        raise ValueError("找不到符合兩年度正常誤報護欄的中風險門檻。")
    medium = max(medium_candidates, key=lambda threshold: (recall(threshold), -threshold))
    high_candidates = [threshold for threshold in candidates if threshold > medium and within_fpr(threshold, high_fpr_limit)]
    if not high_candidates:
        raise ValueError("找不到符合兩年度正常誤報護欄的高風險門檻。")
    high = max(high_candidates, key=lambda threshold: (recall(threshold), -threshold))
    return {"medium_threshold": medium, "high_threshold": high}


def _binary_metrics(records: Sequence[Mapping[str, object]], threshold: float) -> dict[str, float | int]:
    labels = [int(record["label"]) for record in records]
    predictions = [int(float(record["probability"]) >= threshold) for record in records]
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    fraud_precision = tp / (tp + fp) if tp + fp else 0.0
    fraud_recall = tp / (tp + fn) if tp + fn else 0.0
    fraud_f1 = 2 * fraud_precision * fraud_recall / (fraud_precision + fraud_recall) if fraud_precision + fraud_recall else 0.0
    normal_precision = tn / (tn + fn) if tn + fn else 0.0
    normal_recall = tn / (tn + fp) if tn + fp else 0.0
    normal_f1 = 2 * normal_precision * normal_recall / (normal_precision + normal_recall) if normal_precision + normal_recall else 0.0
    return {
        "samples": len(records),
        "fraud_precision": fraud_precision,
        "fraud_recall": fraud_recall,
        "fraud_f1": fraud_f1,
        "macro_f1": (fraud_f1 + normal_f1) / 2,
        "medium_fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def evaluate_candidate(
    records: Sequence[Mapping[str, object]], *, medium_threshold: float, high_threshold: float
) -> dict[str, object]:
    """以未參與調參的 test predictions 執行候選模型硬性驗收。"""
    if not 0 < medium_threshold < high_threshold < 1:
        raise ValueError("候選模型門檻必須滿足 0 < medium < high < 1。")
    normalized = [
        {
            "year": int(record["year"]),
            "label": int(record["label"]),
            "subtype_id": int(record["subtype_id"]),
            "probability": float(record["probability"]),
        }
        for record in records
    ]
    failures: list[str] = []
    years: dict[str, dict[str, float | int]] = {}
    for year in (2022, 2023):
        year_records = [record for record in normalized if record["year"] == year]
        metrics = _binary_metrics(year_records, medium_threshold)
        metrics["high_fpr"] = _binary_metrics(year_records, high_threshold)["medium_fpr"]
        years[str(year)] = metrics
        if metrics["fraud_recall"] < 0.90:
            failures.append(f"{year} 詐騙 Recall 低於 90%。")
        if metrics["medium_fpr"] > 0.12:
            failures.append(f"{year} 中風險正常誤報率高於 12%。")
        if metrics["high_fpr"] > 0.05:
            failures.append(f"{year} 高風險正常誤報率高於 5%。")

    subtypes: dict[str, dict[str, float | int]] = {}
    for subtype_id in sorted({record["subtype_id"] for record in normalized if record["label"] == 1}):
        subtype_positives = [record for record in normalized if record["subtype_id"] == subtype_id]
        subtype_vs_normal = [
            record for record in normalized
            if record["label"] == 0 or record["subtype_id"] == subtype_id
        ]
        metrics = _binary_metrics(subtype_vs_normal, medium_threshold)
        recall = float(metrics["fraud_recall"])
        subtypes[str(subtype_id)] = {
            "samples": len(subtype_positives),
            "recall": recall,
            "precision_vs_normal": metrics["fraud_precision"],
            "f1_vs_normal": metrics["fraud_f1"],
        }
        if len(subtype_positives) >= 50 and recall < 0.70:
            failures.append(f"詐騙子類 {subtype_id} Recall 低於 70%。")

    return {
        "passed": not failures,
        "failures": failures,
        "years": years,
        "overall": _binary_metrics(normalized, medium_threshold),
        "subtypes": subtypes,
    }


def fit_temperature(logits: object, labels: object) -> float:
    """以 validation NLL 擬合單一正溫度參數。"""
    import torch
    from torch.nn import functional as functional

    logits_tensor = torch.as_tensor(logits).detach().float()
    labels_tensor = torch.as_tensor(labels).detach().long()
    if logits_tensor.ndim != 2 or logits_tensor.shape[0] != labels_tensor.numel():
        raise ValueError("校準 logits 必須是 [samples, labels] 且與 labels 等長。")
    if logits_tensor.shape[0] == 0 or set(labels_tensor.tolist()) != {0, 1}:
        raise ValueError("校準資料必須同時包含正常與詐騙樣本。")
    log_temperature = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> object:
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = functional.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 20.0).detach())


def train_candidate(
    prepared_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    base_model: str | PathLike[str],
    base_revision: str | None = None,
    script_view: str = "simplified",
    seed: int = 42,
    max_epochs: int = 8,
    patience: int = 2,
    batch_size: int = 16,
    max_length: int = 256,
    learning_rate: float = 2e-5,
    device: str | None = None,
    checkpoint_interval_steps: int = 500,
    resume_checkpoint: str | PathLike[str] | None = None,
    max_steps: int | None = None,
) -> dict[str, object]:
    """從 prepared records 訓練、校準並驗收一個候選模型。"""
    import json
    from pathlib import Path
    import os
    import shutil
    import time

    import torch
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, get_linear_schedule_with_warmup

    if script_view not in {"simplified", "traditional"}:
        raise ValueError("script_view 必須是 simplified 或 traditional。")
    if not 1 <= max_epochs <= 12 or patience < 1:
        raise ValueError("epochs 必須介於 1–12，且 patience 至少為 1。")
    if checkpoint_interval_steps < 1:
        raise ValueError("checkpoint_interval_steps 至少為 1。")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps 至少為 1。")
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"candidate 輸出目錄不是空的：{output}")
    output.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (prepared / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    splits = {name: [record for record in records if record["split"] == name] for name in ("train", "validation", "test")}
    if any(not split for split in splits.values()):
        raise ValueError("prepared records 必須同時包含 train、validation、test。")
    text_key = "text_simplified" if script_view == "simplified" else "text_traditional"
    torch.manual_seed(seed)
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    base_path = Path(base_model)
    source_kwargs: dict[str, object] = {"local_files_only": base_path.exists()}
    if not base_path.exists() and base_revision is None:
        raise ValueError("遠端 base model 必須固定 base_revision commit SHA。")
    if base_revision is not None:
        source_kwargs["revision"] = base_revision
    tokenizer = AutoTokenizer.from_pretrained(base_model, **source_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=2,
        id2label={0: "NORMAL", 1: "FRAUD"},
        label2id={"NORMAL": 0, "FRAUD": 1},
        ignore_mismatched_sizes=True,
        **source_kwargs,
    ).to(selected_device)

    class EncodedRecords(Dataset):
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self.encodings = tokenizer(
                [str(row[text_key]) for row in rows],
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            self.labels = [int(row["binary_label"]) for row in rows]

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, index: int) -> dict[str, object]:
            item = {name: values[index] for name, values in self.encodings.items()}
            item["labels"] = self.labels[index]
            return item

    train_records = splits["train"]
    train_dataset = EncodedRecords(train_records)
    sampling_weights = compute_sampling_weights(train_records)
    samples_per_epoch = 3 * sum(int(record["binary_label"]) == 1 for record in train_records)
    collator = DataCollatorWithPadding(
        tokenizer,
        padding=True,
        pad_to_multiple_of=8 if selected_device.startswith("cuda") else None,
        return_tensors="pt",
    )

    def make_train_loader(epoch: int) -> DataLoader:
        sampler = WeightedRandomSampler(
            sampling_weights,
            num_samples=samples_per_epoch,
            replacement=True,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        return DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collator)

    validation_loader = DataLoader(EncodedRecords(splits["validation"]), batch_size=batch_size, collate_fn=collator)
    test_loader = DataLoader(EncodedRecords(splits["test"]), batch_size=batch_size, collate_fn=collator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    steps_per_epoch = math.ceil(samples_per_epoch / batch_size)
    total_steps = max_epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.1)),
        num_training_steps=total_steps,
    )
    mixed_precision = selected_device.startswith("cuda")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=mixed_precision)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=mixed_precision)
    )

    def collect(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor, float]:
        model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                labels = batch.pop("labels").to(selected_device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                    logits = model(**{name: value.to(selected_device) for name, value in batch.items()}).logits
                losses.append(float(functional.cross_entropy(logits, labels)))
                all_logits.append(logits.cpu())
                all_labels.append(labels.cpu())
        return torch.cat(all_logits), torch.cat(all_labels), sum(losses) / len(losses)

    def calibrated_records(rows: list[dict[str, object]], logits: torch.Tensor, temperature: float) -> list[dict[str, object]]:
        probabilities = torch.softmax(logits / temperature, dim=-1)[:, 1].tolist()
        return [
            {
                "year": int(row["year"]),
                "label": int(row["binary_label"]),
                "subtype_id": int(row["subtype_id"]),
                "probability": float(probability),
            }
            for row, probability in zip(rows, probabilities)
        ]

    history: list[dict[str, object]] = []
    best_rank: tuple[float, ...] | None = None
    best_epoch = 0
    stale_epochs = 0
    checkpoint_dir = output / "best_checkpoint"
    progress_path = output / "progress_checkpoint.pt"
    start_epoch = 1
    start_step = 0
    global_step = 0
    resumed_from_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_loss_count = 0

    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint)
        state = torch.load(resume_path, map_location=selected_device, weights_only=False)
        expected = {
            "script_view": script_view,
            "seed": seed,
            "base_revision": base_revision,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "max_length": max_length,
        }
        for field, value in expected.items():
            if state.get(field) != value:
                raise ValueError(f"resume checkpoint 的 {field} 不相容。")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"])
        start_step = int(state["next_step"])
        global_step = int(state["global_step"])
        resumed_from_step = global_step
        resumed_epoch_loss_sum = float(state["epoch_loss_sum"])
        resumed_epoch_loss_count = int(state["epoch_loss_count"])
        history = list(state["history"])
        best_rank = state["best_rank"]
        best_epoch = int(state["best_epoch"])
        stale_epochs = int(state["stale_epochs"])
        source_best = resume_path.parent / "best_checkpoint"
        torch.set_rng_state(state["torch_rng_state"].cpu())
        if mixed_precision and state["cuda_rng_state"] is not None:
            torch.cuda.set_rng_state_all(_as_cpu_rng_states(state["cuda_rng_state"]))
        if source_best.is_dir():
            shutil.copytree(source_best, checkpoint_dir)

    def save_progress(
        epoch: int,
        next_step: int,
        epoch_loss_sum: float,
        epoch_loss_count: int,
    ) -> None:
        temporary_path = progress_path.with_suffix(".tmp")
        torch.save(
            {
                "schema_version": 1,
                "script_view": script_view,
                "seed": seed,
                "base_revision": base_revision,
                "max_epochs": max_epochs,
                "batch_size": batch_size,
                "max_length": max_length,
                "epoch": epoch,
                "next_step": next_step,
                "global_step": global_step,
                "epoch_loss_sum": epoch_loss_sum,
                "epoch_loss_count": epoch_loss_count,
                "history": history,
                "best_rank": best_rank,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if mixed_precision else None,
                "scaler": scaler.state_dict(),
            },
            temporary_path,
        )
        os.replace(temporary_path, progress_path)

    training_started = time.perf_counter()
    for epoch in range(start_epoch, max_epochs + 1):
        model.train()
        epoch_loss_sum = resumed_epoch_loss_sum if epoch == start_epoch else 0.0
        epoch_loss_count = resumed_epoch_loss_count if epoch == start_epoch else 0
        train_loader = make_train_loader(epoch)
        for step_index, batch in enumerate(train_loader):
            if epoch == start_epoch and step_index < start_step:
                continue
            labels = batch.pop("labels").to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=mixed_precision):
                logits = model(**{name: value.to(selected_device) for name, value in batch.items()}).logits
                loss = functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_value = float(loss.detach())
            epoch_loss_sum += loss_value
            epoch_loss_count += 1
            global_step += 1
            next_step = step_index + 1
            if global_step % checkpoint_interval_steps == 0:
                save_progress(epoch, next_step, epoch_loss_sum, epoch_loss_count)
            if max_steps is not None and global_step >= max_steps:
                save_progress(epoch, next_step, epoch_loss_sum, epoch_loss_count)
                elapsed_seconds = max(time.perf_counter() - training_started, 1e-9)
                completed_this_run = global_step - resumed_from_step
                progress_manifest = {
                    "status": "paused",
                    "script_view": script_view,
                    "steps_completed": global_step,
                    "steps_per_epoch": steps_per_epoch,
                    "total_steps": total_steps,
                    "steps_per_second": completed_this_run / elapsed_seconds,
                    "estimated_total_hours": total_steps / (completed_this_run / elapsed_seconds) / 3600,
                    "checkpoint": progress_path.name,
                }
                (output / "progress_manifest.json").write_text(
                    json.dumps(progress_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return progress_manifest
        validation_logits, validation_labels, validation_loss = collect(validation_loader)
        temperature = fit_temperature(validation_logits, validation_labels)
        validation_records = calibrated_records(splits["validation"], validation_logits, temperature)
        metrics: dict[str, float | int] | None = None
        try:
            thresholds = select_operating_thresholds(validation_records)
            metrics = _binary_metrics(validation_records, thresholds["medium_threshold"])
            rank = (1.0, float(metrics["fraud_recall"]), float(metrics["macro_f1"]))
            eligible = True
        except ValueError:
            thresholds = None
            rank = (0.0, -validation_loss)
            eligible = False
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss_sum / epoch_loss_count,
                "validation_loss": validation_loss,
                "eligible_thresholds": eligible,
                "fraud_recall": metrics["fraud_recall"] if metrics is not None else None,
                "macro_f1": metrics["macro_f1"] if metrics is not None else None,
                "thresholds": thresholds,
            }
        )
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            stale_epochs = 0
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        else:
            stale_epochs += 1
        save_progress(epoch + 1, 0, 0.0, 0)
        if stale_epochs >= patience:
            break

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir, local_files_only=True).to(selected_device)
    validation_logits, validation_labels, _ = collect(validation_loader)
    temperature = fit_temperature(validation_logits, validation_labels)
    validation_records = calibrated_records(splits["validation"], validation_logits, temperature)
    failures: list[str] = []
    try:
        thresholds = select_operating_thresholds(validation_records)
    except ValueError as exc:
        thresholds = None
        failures.append(str(exc))

    test_report: dict[str, object] | None = None
    candidate_dir: Path | None = None
    if thresholds is not None:
        model.config.id2label = {0: "NORMAL", 1: "FRAUD"}
        model.config.label2id = {"NORMAL": 0, "FRAUD": 1}
        model.config.scamlens_calibration = {"temperature": temperature, **thresholds}
        model.config.scamlens_script_view = script_view
        candidate_dir = output / "model"
        model.save_pretrained(candidate_dir)
        tokenizer.save_pretrained(candidate_dir)
        test_logits, _, _ = collect(test_loader)
        test_records = calibrated_records(splits["test"], test_logits, temperature)
        test_report = evaluate_candidate(test_records, **thresholds)
        failures.extend(test_report["failures"])

    artifact_files: list[dict[str, object]] = []
    if candidate_dir is not None:
        for artifact_path in sorted(path for path in candidate_dir.rglob("*") if path.is_file()):
            digest = hashlib.sha256()
            with artifact_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            artifact_files.append(
                {"path": artifact_path.relative_to(candidate_dir).as_posix(), "sha256": digest.hexdigest(), "size": artifact_path.stat().st_size}
            )

    manifest: dict[str, object] = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "base_model": str(base_model),
        "schema_version": 1,
        "base_revision": base_revision,
        "script_view": script_view,
        "data_manifest": source_manifest,
        "seed": seed,
        "max_epochs": max_epochs,
        "patience": patience,
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "batch_size": batch_size,
        "dynamic_padding": True,
        "mixed_precision": mixed_precision,
        "checkpoint_interval_steps": checkpoint_interval_steps,
        "resumed_from_step": resumed_from_step,
        "steps_completed": global_step,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "max_length": max_length,
        "learning_rate": learning_rate,
        "label_mapping": {"NORMAL": 0, "FRAUD": 1},
        "calibration": {"temperature": temperature, **(thresholds or {})},
        "history": history,
        "optimizer": {"name": "AdamW", "weight_decay": 0.01},
        "scheduler": {"name": "linear_with_warmup", "warmup_ratio": 0.1},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "device": selected_device,
            "cuda_device": torch.cuda.get_device_name(0) if selected_device.startswith("cuda") else None,
        },
        "extension_to_12_epochs_recommended": max_epochs == 8 and len(history) == 8 and best_epoch == 8,
        "artifact_files": artifact_files,
        "test_used_for_selection": False,
        "test_report": test_report,
    }
    (output / "training_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    progress_path.unlink(missing_ok=True)
    return manifest


def select_script_candidate(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """依成對年度／script 指標選出單一模型，或要求補跑 seeds。"""
    normalized = [
        {
            "candidate": str(result["candidate"]),
            "view": str(result["view"]),
            "year": int(result["year"]),
            "seed": int(result["seed"]),
            "fraud_recall": float(result["fraud_recall"]),
            "macro_f1": float(result["macro_f1"]),
        }
        for result in results
    ]
    candidates = {result["candidate"] for result in normalized}
    views = {result["view"] for result in normalized}
    seeds = {result["seed"] for result in normalized}
    if candidates != {"simplified", "traditional"} or views != {"simplified", "traditional"}:
        raise ValueError("選模結果必須同時包含簡體／繁體候選與簡體／繁體 test view。")
    expected = {
        (candidate, view, year, seed)
        for candidate in candidates
        for view in views
        for year in (2022, 2023)
        for seed in seeds
    }
    actual = {(result["candidate"], result["view"], result["year"], result["seed"]) for result in normalized}
    if actual != expected or len(actual) != len(normalized):
        raise ValueError("選模結果缺少成對年度／view／seed 或包含重複項目。")

    comparison_summary: list[dict[str, object]] = []
    for view in sorted(views):
        for year in (2022, 2023):
            for metric in ("fraud_recall", "macro_f1"):
                aggregates: dict[str, dict[str, float]] = {}
                for candidate in sorted(candidates):
                    values = [
                        result[metric]
                        for result in normalized
                        if result["candidate"] == candidate and result["view"] == view and result["year"] == year
                    ]
                    aggregates[candidate] = {"mean": fmean(values), "population_stddev": pstdev(values)}
                comparison_summary.append(
                    {
                        "view": view,
                        "year": year,
                        "metric": metric,
                        "simplified": aggregates["simplified"],
                        "traditional": aggregates["traditional"],
                        "mean_difference": aggregates["simplified"]["mean"] - aggregates["traditional"]["mean"],
                    }
                )

    differences = [float(item["mean_difference"]) for item in comparison_summary]
    if all(difference >= 0.01 for difference in differences):
        return {
            "status": "selected", "candidate": "simplified", "promotion_seed": min(seeds),
            "reason": "significantly_better", "seeds": sorted(seeds), "comparison_summary": comparison_summary,
        }
    if all(difference <= -0.01 for difference in differences):
        return {
            "status": "selected", "candidate": "traditional", "promotion_seed": min(seeds),
            "reason": "significantly_better", "seeds": sorted(seeds), "comparison_summary": comparison_summary,
        }
    if len(seeds) < 3:
        return {
            "status": "additional_seeds_required",
            "candidate": None,
            "reason": "inconclusive",
            "additional_runs_per_candidate": 3 - len(seeds),
            "seeds": sorted(seeds),
            "comparison_summary": comparison_summary,
        }
    equivalent = all(abs(difference) < 0.01 for difference in differences)
    return {
        "status": "selected",
        "candidate": "traditional",
        "promotion_seed": min(seeds),
        "reason": "equivalent" if equivalent else "inconclusive_after_three_seeds",
        "seeds": sorted(seeds),
        "comparison_summary": comparison_summary,
    }


def recalibrate_candidate_artifact(
    model_dir: str | PathLike[str],
    prepared_dir: str | PathLike[str],
    output_dir: str | PathLike[str],
    *,
    script_view: str,
    batch_size: int = 64,
    max_length: int = 256,
    device: str | None = None,
    medium_fpr_limit: float = 0.08,
    high_fpr_limit: float = 0.03,
) -> dict[str, object]:
    """只用 validation 重算既有模型校準值，BERT 權重保持不變。"""
    import json
    from pathlib import Path
    import shutil

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if script_view not in {"simplified", "traditional"}:
        raise ValueError("script_view 必須是 simplified 或 traditional。")
    model_path = Path(model_dir)
    prepared = Path(prepared_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"recalibration 輸出目錄已存在：{output}")
    weight_paths = sorted(model_path.glob("*.safetensors")) or sorted(model_path.glob("pytorch_model*.bin"))
    if not weight_paths:
        raise FileNotFoundError("candidate model 缺少 BERT 權重檔。")

    rows = [
        json.loads(line)
        for line in (prepared / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["split"] == "validation"]
    if not rows:
        raise ValueError("prepared records 缺少 validation split。")

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(selected_device)
    id2label = {int(key): str(value).upper() for key, value in model.config.id2label.items()}
    label2id = {str(key).upper(): int(value) for key, value in model.config.label2id.items()}
    if id2label != {0: "NORMAL", 1: "FRAUD"} or label2id != {"NORMAL": 0, "FRAUD": 1}:
        raise ValueError("BERT 模型標籤契約必須是 0=NORMAL、1=FRAUD。")

    text_key = "text_simplified" if script_view == "simplified" else "text_traditional"
    logits_parts: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [str(row[text_key]) for row in batch],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits_parts.append(
                model(**{name: value.to(selected_device) for name, value in encoded.items()}).logits.cpu()
            )
    logits = torch.cat(logits_parts)
    labels = torch.tensor([int(row["binary_label"]) for row in rows])
    temperature = fit_temperature(logits, labels)
    probabilities = torch.softmax(logits / temperature, dim=-1)[:, 1].tolist()
    validation_records = [
        {
            "year": int(row["year"]),
            "label": int(row["binary_label"]),
            "probability": float(probability),
        }
        for row, probability in zip(rows, probabilities)
    ]
    thresholds = select_operating_thresholds(
        validation_records,
        medium_fpr_limit=medium_fpr_limit,
        high_fpr_limit=high_fpr_limit,
    )

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    before = {path.name: sha256_file(path) for path in weight_paths}
    shutil.copytree(model_path, output)
    model.config.scamlens_calibration = {"temperature": temperature, **thresholds}
    model.config.scamlens_script_view = script_view
    model.config.save_pretrained(output)
    after = {path.name: sha256_file(output / path.name) for path in weight_paths}
    if before != after:
        raise RuntimeError("重新校準不得改變 BERT 權重。")

    manifest = {
        "status": "passed",
        "script_view": script_view,
        "validation_samples": len(rows),
        "test_used_for_calibration": False,
        "calibration": model.config.scamlens_calibration,
        "validation_fpr_limits": {"medium": medium_fpr_limit, "high": high_fpr_limit},
        "weights_sha256_before": next(iter(before.values())),
        "weights_sha256_after": next(iter(after.values())),
        "weight_artifacts": [
            {"path": name, "sha256": digest}
            for name, digest in before.items()
        ],
    }
    (output / "recalibration_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest

def evaluate_candidate_artifact(
    model_dir: str | PathLike[str],
    prepared_dir: str | PathLike[str],
    *,
    script_view: str,
    batch_size: int = 64,
    max_length: int = 256,
    device: str | None = None,
) -> dict[str, object]:
    """在指定 script test view 上獨立驗收已校準的 candidate artifact。"""
    import json
    from pathlib import Path

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.bert_runtime import BertRuntime

    if script_view not in {"simplified", "traditional"}:
        raise ValueError("script_view 必須是 simplified 或 traditional。")
    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(selected_device)
    runtime = BertRuntime(tokenizer, model, device=selected_device)
    calibration = model.config.scamlens_calibration
    temperature = float(calibration["temperature"])
    medium_threshold = float(calibration["medium_threshold"])
    high_threshold = float(calibration["high_threshold"])
    text_key = "text_simplified" if script_view == "simplified" else "text_traditional"
    prepared = Path(prepared_dir)
    rows = [
        json.loads(line)
        for line in (prepared / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row["split"] == "test"]
    if not rows:
        raise ValueError("prepared records 缺少 test split。")
    probabilities: list[float] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch = rows[start:start + batch_size]
            encoded = tokenizer(
                [str(row[text_key]) for row in batch],
                max_length=max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            logits = model(**{name: value.to(selected_device) for name, value in encoded.items()}).logits
            probabilities.extend(torch.softmax(logits / temperature, dim=-1)[:, runtime.fraud_label_id].cpu().tolist())
    evaluation_records = [
        {
            "year": int(row["year"]),
            "label": int(row["binary_label"]),
            "subtype_id": int(row["subtype_id"]),
            "probability": float(probability),
        }
        for row, probability in zip(rows, probabilities)
    ]
    report = evaluate_candidate(
        evaluation_records,
        medium_threshold=medium_threshold,
        high_threshold=high_threshold,
    )
    report["script_view"] = script_view
    return report
