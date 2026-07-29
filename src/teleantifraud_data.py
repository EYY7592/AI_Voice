"""將 TeleAntiFraud ASR 文字安全加入 ChiFraud 訓練集。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import unicodedata

import opencc


SOURCE_URL = "https://www.kaggle.com/datasets/divyanshsharma23/research28k"
CANONICAL_URL = "https://huggingface.co/datasets/JimmyMa99/TeleAntiFraud"
PAPER_URL = "https://arxiv.org/abs/2503.24115"
_LABEL_RE = re.compile(r'"is_fraud"\s*:\s*(true|false)', re.IGNORECASE)
_PROMPT_MARKERS = (
    "\n\n根据听到的音频内容",
    "\n\n根据你听到的音频内容",
    "\n\n**任务描述：**",
    "\n\n任务描述：",
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def _parse_row(row: object, line_number: int) -> tuple[str, int]:
    if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
        raise ValueError(f"第 {line_number} 行缺少 messages。")
    messages = row["messages"]
    if len(messages) < 2 or messages[0].get("role") != "user":
        raise ValueError(f"第 {line_number} 行不是預期的多輪訊息格式。")
    content = str(messages[0].get("content", ""))
    if not content.startswith("音频内容："):
        raise ValueError(f"第 {line_number} 行缺少音频内容前綴。")
    text = content[len("音频内容：") :]
    marker_positions = [text.find(marker) for marker in _PROMPT_MARKERS if text.find(marker) >= 0]
    if not marker_positions:
        raise ValueError(f"第 {line_number} 行找不到音訊文字與任務提示的邊界。")
    text = _normalize(text[: min(marker_positions)])
    if not text:
        raise ValueError(f"第 {line_number} 行的音訊文字為空。")

    labels: list[int] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        matches = _LABEL_RE.findall(str(message.get("content", "")))
        if matches:
            labels.append(int(matches[-1].lower() == "true"))
    if not labels:
        raise ValueError(f"第 {line_number} 行找不到 is_fraud 標籤。")
    if len(set(labels)) != 1:
        raise ValueError(f"第 {line_number} 行包含互相衝突的 is_fraud 標籤。")
    return text, labels[0]


def prepare_teleantifraud_augmentation(
    source_path: str | Path,
    base_prepared_dir: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
    seed: int = 42,
) -> dict[str, object]:
    """只增強 train；保留 ChiFraud validation/test，另存非 gold 診斷集。"""
    if not source_revision.strip():
        raise ValueError("source_revision 不得為空。")
    source = Path(source_path)
    base = Path(base_prepared_dir)
    output = Path(output_dir)
    if not source.is_file():
        raise FileNotFoundError(source)
    base_manifest_path = base / "manifest.json"
    base_records_path = base / "records.jsonl"
    if not base_manifest_path.is_file() or not base_records_path.is_file():
        raise FileNotFoundError("base_prepared_dir 缺少 manifest.json 或 records.jsonl。")

    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    base_records = [
        json.loads(line) for line in base_records_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    converter = opencc.OpenCC("s2twp")
    base_texts = {_normalize(str(row["text_traditional"])) for row in base_records}
    records_by_text: dict[str, dict[str, object]] = {}
    conflicts: set[str] = set()
    raw_records = 0

    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            raw_records += 1
            try:
                text_simplified, label = _parse_row(json.loads(line), line_number)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_number} 行不是有效 JSON。") from exc
            text_traditional = _normalize(converter.convert(text_simplified))
            if text_traditional in base_texts or text_traditional in conflicts:
                continue
            existing = records_by_text.get(text_traditional)
            if existing is not None:
                if existing["binary_label"] != label:
                    del records_by_text[text_traditional]
                    conflicts.add(text_traditional)
                continue
            record_id = hashlib.sha256(f"TeleAntiFraud-28k\0{label}\0{text_simplified}".encode()).hexdigest()
            records_by_text[text_traditional] = {
                "record_id": record_id,
                "year": 2025,
                "subtype_id": 11,
                "subtype": "TeleAntiFraud",
                "binary_label": label,
                "text_simplified": text_simplified,
                "text_traditional": text_traditional,
                "source_dataset": "TeleAntiFraud-28k",
            }

    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for record in records_by_text.values():
        groups[int(record["binary_label"])].append(record)
    if set(groups) != {0, 1}:
        raise ValueError("增強資料必須同時包含正常與詐騙樣本。")

    augmentation_train: list[dict[str, object]] = []
    diagnostic: list[dict[str, object]] = []
    for label, group in groups.items():
        group.sort(key=lambda item: _stable_order(str(item["record_id"]), seed))
        train_end = int(len(group) * 0.8)
        validation_end = train_end + int(len(group) * 0.1)
        for index, record in enumerate(group):
            if index < train_end:
                record["split"] = "train"
                augmentation_train.append(record)
            else:
                record["split"] = "diagnostic_validation" if index < validation_end else "diagnostic_test"
                diagnostic.append(record)

    combined = sorted(base_records + augmentation_train, key=lambda item: str(item["record_id"]))
    diagnostic.sort(key=lambda item: str(item["record_id"]))
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in combined),
        encoding="utf-8",
    )
    (output / "augmentation_diagnostic.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in diagnostic),
        encoding="utf-8",
    )

    split_id_sha256 = {
        split: hashlib.sha256(
            (
                "\n".join(sorted(str(row["record_id"]) for row in combined if row["split"] == split)) + "\n"
            ).encode()
        ).hexdigest()
        for split in ("train", "validation", "test")
    }
    counts = Counter((str(row["split"]), int(row["binary_label"])) for row in records_by_text.values())
    manifest: dict[str, object] = {
        "dataset": "ChiFraud+TeleAntiFraud-28k",
        "schema_version": 1,
        "seed": seed,
        "base_dataset": base_manifest,
        "split_id_sha256": split_id_sha256,
        "augmentation": {
            "source_url": SOURCE_URL,
            "canonical_url": CANONICAL_URL,
            "paper_url": PAPER_URL,
            "source_revision": source_revision,
            "source_sha256": _sha256(source),
            "license": "Apache-2.0",
            "raw_records": raw_records,
            "deduplicated_records": len(records_by_text),
            "excluded_duplicate_overlap_or_conflict_records": raw_records - len(records_by_text),
            "conflicting_texts_dropped": len(conflicts),
            "counts": [
                {"split": split, "binary_label": label, "count": count}
                for (split, label), count in sorted(counts.items())
            ],
            "generation": "hybrid_real_asr_llm_tts_multi_agent",
            "diagnostic_only": True,
            "taiwan_gold": False,
        },
        "opencc": {"config": "s2twp", "version": version("opencc-python-reimplemented")},
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="將 TeleAntiFraud ASR 文字加入 ChiFraud train")
    parser.add_argument("source_path", type=Path)
    parser.add_argument("base_prepared_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = prepare_teleantifraud_augmentation(
        args.source_path,
        args.base_prepared_dir,
        args.output_dir,
        source_revision=args.source_revision,
        seed=args.seed,
    )
    print(json.dumps(manifest["augmentation"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
