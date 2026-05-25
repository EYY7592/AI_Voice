"""
Privacy regression tests for API history and persistent memory metadata.
"""
import numpy as np

from src.privacy import build_private_case_metadata, summarize_history_metadata
from src.step3_agents.memory_agent import MemoryAgent


def test_history_summary_does_not_return_raw_transcript():
    raw_text = "請提供你的銀行帳號與身分證字號，並轉入安全帳戶"
    rows = summarize_history_metadata([
        {
            "id": "case-1",
            "timestamp": "2026-05-26 10:00:00",
            "fraud_type": "高風險",
            "text": raw_text,
            "confirmed": True,
        }
    ])

    assert rows[0]["text"] != raw_text
    assert "銀行帳號" not in rows[0]["text"]
    assert rows[0]["text_length"] == len(raw_text)


def test_private_case_metadata_does_not_store_raw_transcript():
    raw_text = "這是一段不能寫進長期記憶 metadata 的完整通話文字"
    metadata = build_private_case_metadata("case-2", raw_text, "中風險")

    assert "text" not in metadata
    assert metadata["text_length"] == len(raw_text)
    assert len(metadata["text_sha256"]) == 64


def test_history_summary_filters_deleted_records():
    rows = summarize_history_metadata([
        {"id": "deleted", "text": "secret", "_deleted": True},
        {"id": "active", "text_length": 3, "fraud_type": "低風險"},
    ])

    assert [row["id"] for row in rows] == ["active"]


def test_delete_case_handles_metadata_without_raw_text():
    agent = MemoryAgent()
    memory = agent.persistent_memory
    memory._metadata = [
        {"id": "case-1", "text_length": 5, "fraud_type": "高風險"},
        {"id": "case-2", "text_length": 6, "fraud_type": "中風險"},
    ]

    class FakeIndex:
        ntotal = 2

        def search(self, query, k):
            return (
                np.array([[0.9, 0.8]], dtype=np.float32),
                np.array([[0, 1]], dtype=np.int64),
            )

    memory._index = FakeIndex()
    memory.save = lambda: None

    assert agent.delete_case("case-1") is True
    assert memory._metadata[0]["_deleted"] is True
