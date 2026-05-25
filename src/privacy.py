"""
Privacy helpers for transcript handling.

The web API can show the current analysis result to the current caller, but
long-term history and logs must not persist or expose raw transcript content.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def text_sha256(text: str) -> str:
    """Return a stable fingerprint without storing the original text."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def redacted_text_label(text_length: int | None) -> str:
    """Build a UI-safe placeholder for transcript text."""
    if text_length is None:
        return "文字已隱藏"
    return f"文字已隱藏（長度 {text_length} 字）"


def build_private_case_metadata(
    case_id: str,
    text: str,
    fraud_type: str,
    confirmed: bool = True,
) -> dict:
    """Create persistent metadata without raw transcript content."""
    text = text or ""
    metadata = {
        "text_length": len(text),
        "text_sha256": text_sha256(text),
        "fraud_type": fraud_type,
        "confirmed": confirmed,
    }
    if case_id:
        metadata["id"] = case_id
    return metadata


def sanitize_history_item(meta: dict) -> dict:
    """Return an API-safe history record."""
    raw_text = meta.get("text", "")
    text_length = meta.get("text_length")
    if text_length is None and raw_text:
        text_length = len(raw_text)

    return {
        "id": meta.get("id", ""),
        "timestamp": meta.get("timestamp", ""),
        "fraud_type": meta.get("fraud_type", "未知"),
        "confirmed": bool(meta.get("confirmed", False)),
        "text": redacted_text_label(text_length),
        "text_length": text_length or 0,
    }


def summarize_history_metadata(metadata: Iterable[dict]) -> list[dict]:
    """Filter deleted records and redact transcript fields for API history."""
    return [
        sanitize_history_item(meta)
        for meta in metadata
        if not meta.get("_deleted")
    ]


def normalize_persistent_metadata(meta: dict) -> dict:
    """Strip raw transcript content from metadata loaded from older files."""
    if "text" not in meta:
        return meta

    raw_text = meta.pop("text") or ""
    meta.setdefault("text_length", len(raw_text))
    meta.setdefault("text_sha256", text_sha256(raw_text))
    return meta


def safe_upload_suffix(filename: str | None) -> str:
    """Keep only a harmless upload suffix for tempfile creation and logging."""
    suffix = Path(filename or "audio.wav").suffix.lower()
    if not suffix or len(suffix) > 10:
        return ".wav"
    return suffix
