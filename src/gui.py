"""ScamLens-TW localhost Web API。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from src.bert_runtime import BertRuntime
from src.media_extractors import (
    AUDIO_SUFFIXES,
    IMAGE_SUFFIXES,
    MAX_AUDIO_BYTES,
    MAX_IMAGE_BYTES,
    AudioTextExtractor,
    EasyOcrReader,
)
from src.scam_analysis import ScamAnalyzer
from src.text_correction import TextCorrector
from src.utils.logger import setup_logger


ROOT_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = ROOT_DIR / "static"
DISCLAIMER = "此分數是未經台灣真實 gold set 校準的風險指標，不是詐騙機率或事實認定。"
IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/webp"}
AUDIO_MIME_TYPES = {
    "audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp4", "audio/x-m4a",
    "audio/flac", "audio/x-flac", "audio/ogg", "application/ogg",
}
logger = setup_logger("scamlens.gui", level="INFO")


class LazyBertRuntime:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self._runtime: BertRuntime | None = None

    def predict_details(self, text: str) -> dict[str, object]:
        if self._runtime is None:
            self._runtime = BertRuntime.load(self.model_path)
        return self._runtime.predict_details(text)


def create_app(
    *,
    bert_runtime: Any | None = None,
    corrector: Any | None = None,
    image_reader: Any | None = None,
    audio_reader: Any | None = None,
) -> FastAPI:
    app = FastAPI(title="ScamLens-TW 通用型防詐檢測工具", version="4.0.0")
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    analyzer = ScamAnalyzer(
        medium_risk_score=settings.analysis.medium_risk_score,
        high_risk_score=settings.analysis.high_risk_score,
    )

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(
            (STATIC_DIR / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.post("/api/analyze")
    async def analyze(
        text: str | None = Form(default=None),
        upload: UploadFile | None = File(default=None),
        correction_confirmed: bool = Form(default=False),
        source_type: str | None = Form(default=None),
    ) -> JSONResponse:
        provided = int(bool(text and text.strip())) + int(upload is not None)
        if provided != 1:
            return JSONResponse(
                {"detail": "一次必須且只能提供文字、圖片或語音其中一種輸入。"},
                status_code=422,
            )

        started = time.perf_counter()
        input_type = source_type if source_type in {"image", "audio"} else "text"
        original_text = text or ""
        extraction: dict[str, object] = {}
        if upload is not None:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                allowed_mime_types = IMAGE_MIME_TYPES
                max_bytes = MAX_IMAGE_BYTES
            elif suffix in AUDIO_SUFFIXES:
                allowed_mime_types = AUDIO_MIME_TYPES
                max_bytes = MAX_AUDIO_BYTES
            else:
                return JSONResponse({"detail": "不支援的檔案格式。"}, status_code=415)
            if upload.content_type not in allowed_mime_types:
                return JSONResponse({"detail": "副檔名與媒體類型不相符。"}, status_code=415)
            content = await upload.read(max_bytes + 1)
            if len(content) > max_bytes:
                return JSONResponse({"detail": "上傳檔案超過大小限制。"}, status_code=413)
            try:
                if suffix in IMAGE_SUFFIXES:
                    if image_reader is None:
                        return JSONResponse({"detail": "本機 OCR 模型尚未準備。"}, status_code=503)
                    input_type = "image"
                    extraction = image_reader.extract(content)
                elif suffix in AUDIO_SUFFIXES:
                    if audio_reader is None:
                        return JSONResponse({"detail": "本機 Whisper 模型尚未準備。"}, status_code=503)
                    input_type = "audio"
                    extraction = audio_reader.extract(content, suffix)
            except ValueError as exc:
                return JSONResponse({"detail": str(exc)}, status_code=413)
            except Exception as exc:
                logger.warning("%s 文字擷取不可用：%s", input_type, type(exc).__name__)
                return JSONResponse({"detail": f"本機 {input_type} 文字擷取失敗。"}, status_code=503)
            original_text = str(extraction.pop("text", ""))

        if len(original_text) > 20_000:
            return JSONResponse({"detail": "文字不可超過 20,000 字。"}, status_code=413)

        analysis_text = original_text
        correction_status = "confirmed" if correction_confirmed else "disabled"
        corrections: list[dict[str, object]] = []
        if not correction_confirmed and corrector is not None:
            proposal = corrector.suggest(analysis_text)
            suggested_text = str(proposal["suggested_text"])
            corrections = list(proposal.get("changes", []))
            correction_status = str(proposal.get("model_status", "unavailable"))
            if suggested_text != analysis_text:
                return JSONResponse({
                    "status": "needs_confirmation",
                    "input_type": input_type,
                    "original_text": original_text,
                    "suggested_text": suggested_text,
                    "analysis_text": None,
                    "corrections": corrections,
                    "extraction": extraction,
                    "risk_score": None,
                    "risk_level": "等待確認",
                    "categories": [],
                    "evidence": [],
                    "safety_actions": [],
                    "analysis_windows": 0,
                    "bert_evidence": [],
                    "model_status": {"bert": "not_run", "correction": correction_status},
                    "elapsed": round(time.perf_counter() - started, 3),
                    "disclaimer": DISCLAIMER,
                })
            analysis_text = suggested_text

        bert_details: dict[str, object] = {
            "probability": None,
            "window_count": 0,
            "highest_risk_windows": [],
        }
        bert_status = "disabled"
        if bert_runtime is not None:
            try:
                bert_details = bert_runtime.predict_details(analysis_text)
                bert_status = "ready"
            except Exception as exc:
                logger.warning("BERT 輔助分析不可用：%s", type(exc).__name__)
                bert_status = "unavailable"

        probability = bert_details.get("probability")
        result = analyzer.analyze_text(
            analysis_text,
            bert_probability=float(probability) if probability is not None else None,
            window_count=int(bert_details.get("window_count", 0)),
        )
        return JSONResponse({
            "status": result.status,
            "input_type": input_type,
            "original_text": original_text,
            "analysis_text": analysis_text,
            "corrections": corrections,
            "extraction": extraction,
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
            "categories": result.categories,
            "evidence": result.evidence,
            "safety_actions": result.safety_actions,
            "analysis_windows": result.window_count,
            "bert_evidence": bert_details.get("highest_risk_windows", []),
            "model_status": {"bert": bert_status, "correction": correction_status},
            "elapsed": round(time.perf_counter() - started, 3),
            "disclaimer": DISCLAIMER,
        })

    return app


app = create_app(
    bert_runtime=LazyBertRuntime(settings.models.bert),
    corrector=TextCorrector(settings.models.correction),
    image_reader=EasyOcrReader(settings.models.ocr),
    audio_reader=AudioTextExtractor(model_path=settings.models.whisper),
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=7861)
