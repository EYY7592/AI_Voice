"""可重現的 ChiFraud 簡繁成對資料準備。"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re
import unicodedata

import opencc


CHIFRAUD_SOURCE_URL = "https://github.com/xuemingxxx/ChiFraud"
SOURCE_FILES = {
    "ChiFraud_train.csv": 2022,
    "ChiFraud_t2022.csv": 2022,
    "ChiFraud_t2023.csv": 2023,
}
SUBTYPES = {
    0: "Normal",
    1: "Gambling",
    2: "Whoring",
    3: "Credentials",
    4: "Bank",
    5: "Drugs",
    6: "Cash-out",
    7: "Certification",
    8: "SIM",
    9: "Loan",
    10: "New",
}
SOURCE_CLASSES = {
    0: "正常",
    1: "赌博博彩",
    2: "招嫖色情",
    3: "办假证",
    4: "虚假办卡",
    5: "违禁药品交易",
    6: "违规提现",
    7: "虚假证明",
    8: "虚假手机卡",
    9: "地下黑贷",
    10: "新类型",
}



def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_order(record_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()


def prepare_chifraud_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    source_revision: str,
    seed: int = 42,
) -> dict[str, object]:
    """稽核、去重並輸出同 split 的簡體與台灣繁體資料。"""
    if not source_revision.strip():
        raise ValueError("source_revision 不得為空。")
    source = Path(source_dir)
    output = Path(output_dir)
    class_path = source / "class.txt"
    if not class_path.is_file():
        raise FileNotFoundError("缺少 ChiFraud 分類清單：class.txt")
    source_classes: dict[int, str] = {}
    for line_number, line in enumerate(class_path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"class.txt:{line_number} 格式錯誤。")
        try:
            source_classes[int(parts[0])] = parts[1]
        except ValueError as exc:
            raise ValueError(f"class.txt:{line_number} 的標籤不是整數。") from exc
    if source_classes != SOURCE_CLASSES:
        raise ValueError("class.txt 與預期的 ChiFraud 0–10 分類清單不一致。")
    converter = opencc.OpenCC("s2twp")
    records_by_text: dict[str, dict[str, object]] = {}
    conflicting_texts: set[str] = set()
    conflicting_records_dropped = 0
    files: list[dict[str, object]] = [{"name": "class.txt", "sha256": _sha256_bytes(class_path)}]
    raw_records = 0

    for filename, year in SOURCE_FILES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少 ChiFraud 資料檔：{filename}")
        file_records = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                if line_number == 1 and tuple(column.strip() for column in row[:2]) == ("Label_id", "Text"):
                    continue
                if len(row) < 2:
                    raise ValueError(f"{filename}:{line_number} 缺少標籤或文字。")
                try:
                    subtype_id = int(row[0].strip())
                except ValueError as exc:
                    raise ValueError(f"{filename}:{line_number} 的標籤不是整數。") from exc
                if subtype_id not in SUBTYPES:
                    raise ValueError(f"{filename}:{line_number} 包含未知標籤 {subtype_id}。")
                text_simplified = _normalize("\t".join(row[1:]))
                if not text_simplified:
                    raise ValueError(f"{filename}:{line_number} 的文字為空。")
                text_traditional = _normalize(converter.convert(text_simplified))
                dedup_key = text_traditional
                existing = records_by_text.get(dedup_key)
                if dedup_key in conflicting_texts:
                    conflicting_records_dropped += 1
                elif existing is not None:
                    if existing["subtype_id"] != subtype_id:
                        del records_by_text[dedup_key]
                        conflicting_texts.add(dedup_key)
                        conflicting_records_dropped += 2
                else:
                    record_id = hashlib.sha256(f"{subtype_id}\0{text_simplified}".encode()).hexdigest()
                    records_by_text[dedup_key] = {
                        "record_id": record_id,
                        "year": year,
                        "subtype_id": subtype_id,
                        "subtype": SUBTYPES[subtype_id],
                        "binary_label": int(subtype_id != 0),
                        "text_simplified": text_simplified,
                        "text_traditional": text_traditional,
                    }
                raw_records += 1
                file_records += 1
        files.append({"name": filename, "year": year, "sha256": _sha256_bytes(path), "records": file_records})

    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for record in records_by_text.values():
        groups[(int(record["year"]), int(record["subtype_id"]))].append(record)
    for group in groups.values():
        group.sort(key=lambda item: _stable_order(str(item["record_id"]), seed))
        train_end = int(len(group) * 0.8)
        validation_end = train_end + int(len(group) * 0.1)
        for index, record in enumerate(group):
            record["split"] = "train" if index < train_end else "validation" if index < validation_end else "test"

    records = sorted(records_by_text.values(), key=lambda item: str(item["record_id"]))
    counts = Counter((int(item["year"]), str(item["split"]), str(item["subtype"])) for item in records)
    output.mkdir(parents=True, exist_ok=True)
    (output / "records.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records),
        encoding="utf-8",
    )
    split_id_sha256 = {
        split: hashlib.sha256(
            ("\n".join(sorted(str(item["record_id"]) for item in records if item["split"] == split)) + "\n").encode()
        ).hexdigest()
        for split in ("train", "validation", "test")
    }
    manifest: dict[str, object] = {
        "dataset": "ChiFraud",
        "source_revision": source_revision,
        "files": files,
        "schema_version": 1,
        "source_url": CHIFRAUD_SOURCE_URL,
        "seed": seed,
        "split_ratio": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "opencc": {"config": "s2twp", "version": version("opencc-python-reimplemented")},
        "raw_records": raw_records,
        "deduplicated_records": len(records),
        "split_id_sha256": split_id_sha256,
        "duplicate_records": raw_records - len(records),
        "conflicting_texts": len(conflicting_texts),
        "conflicting_records_dropped": conflicting_records_dropped,
        "counts": [
            {"year": year, "split": split, "subtype": subtype, "count": count}
            for (year, split, subtype), count in sorted(counts.items())
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="準備可重現的 ChiFraud 簡繁成對資料")
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    manifest = prepare_chifraud_dataset(
        args.source_dir, args.output_dir, source_revision=args.source_revision, seed=args.seed
    )
    print(json.dumps({"output": str(args.output_dir), "records": manifest["deduplicated_records"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
