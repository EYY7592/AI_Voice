import hashlib
import json

from src.teleantifraud_data import prepare_teleantifraud_augmentation


def test_prepare_teleantifraud_augmentation_keeps_base_evaluation_untouched(tmp_path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    base_records = [
        {
            "record_id": f"base-{split}",
            "year": 2022,
            "subtype_id": 0,
            "subtype": "Normal",
            "binary_label": 0,
            "text_simplified": split,
            "text_traditional": split,
            "split": split,
        }
        for split in ("train", "validation", "test")
    ]
    (base / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in base_records),
        encoding="utf-8",
    )
    (base / "manifest.json").write_text(
        json.dumps({"dataset": "ChiFraud", "source_revision": "fixture"}),
        encoding="utf-8",
    )

    source = tmp_path / "asr.jsonl"
    rows = []
    for label in (False, True):
        for index in range(10):
            transcript = f"{'可疑匯款' if label else '正常訂餐'}對話 {index}"
            rows.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"音频内容： {transcript}\n\n根据听到的音频内容，分析该通话是否涉及诈骗。",
                        },
                        {
                            "role": "assistant",
                            "content": f'<answer>{{"is_fraud": {str(label).lower()}}}</answer>',
                        },
                    ]
                }
            )
    rows.append(rows[0])
    source.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    output = tmp_path / "prepared"
    manifest = prepare_teleantifraud_augmentation(
        source,
        base,
        output,
        source_revision="fixture-v1",
        seed=42,
    )

    combined = [
        json.loads(line)
        for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    diagnostic = [
        json.loads(line)
        for line in (output / "augmentation_diagnostic.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {
        row["record_id"]: row for row in combined if row["split"] in {"validation", "test"}
    } == {row["record_id"]: row for row in base_records[1:]}
    assert len([row for row in combined if row.get("source_dataset") == "TeleAntiFraud-28k"]) == 16
    assert len(diagnostic) == 4
    assert {row["split"] for row in diagnostic} == {"diagnostic_validation", "diagnostic_test"}
    assert len({row["record_id"] for row in combined + diagnostic}) == len(combined) + len(diagnostic)
    assert manifest["augmentation"]["raw_records"] == 21
    assert manifest["augmentation"]["deduplicated_records"] == 20
    assert manifest["augmentation"]["taiwan_gold"] is False
    assert manifest["augmentation"]["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
