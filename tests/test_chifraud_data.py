import json
import subprocess
import sys
import pytest

from src.chifraud_data import prepare_chifraud_dataset


CLASS_TEXT = """0 正常
1 赌博博彩
2 招嫖色情
3 办假证
4 虚假办卡
5 违禁药品交易
6 违规提现
7 虚假证明
8 虚假手机卡
9 地下黑贷
10 新类型
"""


def _write_classes(path) -> None:
    path.write_text(CLASS_TEXT, encoding="utf-8")


def _write_rows(path, prefix: str, *, include_official_header: bool = False) -> None:
    rows = []
    if include_official_header:
        rows.append("Label_id\tText\n")
    for label in (0, 1):
        for index in range(10):
            rows.append(f"{label}\t{prefix}软件内容 {label}-{index}\n")
    path.write_text("".join(rows), encoding="utf-8")


def test_prepare_chifraud_dataset_produces_paired_stratified_splits(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    _write_rows(source / "ChiFraud_train.csv", "train")
    _write_rows(source / "ChiFraud_t2022.csv", "test2022")
    _write_rows(source / "ChiFraud_t2023.csv", "test2023")
    _write_classes(source / "class.txt")
    with (source / "ChiFraud_t2022.csv").open("a", encoding="utf-8") as handle:
        handle.write("1\t共同重复内容\n")
    with (source / "ChiFraud_t2023.csv").open("a", encoding="utf-8") as handle:
        handle.write("1\t共同重复内容\n")

    output = tmp_path / "prepared"
    manifest = prepare_chifraud_dataset(source, output, source_revision="fixture-v1", seed=42)

    records = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["source_revision"] == "fixture-v1"
    assert manifest["raw_records"] == 62
    assert manifest["deduplicated_records"] == 61
    assert len({record["record_id"] for record in records}) == 61
    assert {record["year"] for record in records} == {2022, 2023}
    assert {record["split"] for record in records if record["year"] == 2022} == {"train", "validation", "test"}
    assert {record["split"] for record in records if record["year"] == 2023} == {"train", "validation", "test"}
    assert all(record["text_simplified"] and record["text_traditional"] for record in records)
    assert any("軟體" in record["text_traditional"] for record in records)


def test_prepare_chifraud_dataset_accepts_official_file_header(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    _write_rows(source / "ChiFraud_train.csv", "train", include_official_header=True)
    _write_rows(source / "ChiFraud_t2022.csv", "test2022", include_official_header=True)
    _write_rows(source / "ChiFraud_t2023.csv", "test2023", include_official_header=True)
    _write_classes(source / "class.txt")

    output = tmp_path / "prepared"
    manifest = prepare_chifraud_dataset(source, output, source_revision="fixture-v1", seed=42)

    records = [json.loads(line) for line in (output / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert manifest["raw_records"] == 60
    assert all(record["text_simplified"] != "Text" for record in records)


def test_prepare_chifraud_dataset_rejects_incomplete_class_list(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    _write_rows(source / "ChiFraud_train.csv", "train")
    _write_rows(source / "ChiFraud_t2022.csv", "test2022")
    _write_rows(source / "ChiFraud_t2023.csv", "test2023")
    bad_classes = CLASS_TEXT.replace("10 新类型\n", "")
    (source / "class.txt").write_text(bad_classes, encoding="utf-8")

    with pytest.raises(ValueError, match="class.txt"):
        prepare_chifraud_dataset(source, tmp_path / "prepared", source_revision="fixture-v1")


def test_chifraud_data_cli_uses_the_same_preparation_flow(tmp_path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    _write_rows(source / "ChiFraud_train.csv", "train")
    _write_rows(source / "ChiFraud_t2022.csv", "test2022")
    _write_rows(source / "ChiFraud_t2023.csv", "test2023")
    _write_classes(source / "class.txt")
    output = tmp_path / "prepared"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.chifraud_data",
            str(source),
            str(output),
            "--source-revision",
            "fixture-v1",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["source_revision"] == "fixture-v1"
