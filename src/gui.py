"""
AI_Voice 智慧語音詐騙檢測 — Web Server (FastAPI)
==================================================
提供 REST API 和靜態檔案伺服，作為自定義 Web UI 的後端。
"""
import json
import time
import tempfile
import traceback
import queue
import threading
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from src.step1_preprocessing.audio_loader import AudioLoader
from src.step1_preprocessing.denoiser import Denoiser
from src.step1_preprocessing.feature_extractor import FeatureExtractor
from src.models import (
    AgentResult,
    FusionResult,
    TranscriptionResult,
    TranscriptionSegment,
)
from src.utils.logger import setup_logger
from config.settings import settings
from src.step2_transcription.whisper_transcriber import WhisperTranscriber
from src.step3_agents.semantic_agent import SemanticAgent
from src.step3_agents.voiceprint_agent import VoiceprintAgent
from src.step3_agents.memory_agent import MemoryAgent
from src.step4_fusion.se_attention_fusion import SEAttentionFusion
from src.privacy import (
    build_private_case_metadata,
    safe_upload_suffix,
    summarize_history_metadata,
)

logger = setup_logger("ai_voice.gui", level="INFO")

# ========== 初始化模組 ==========
audio_loader = AudioLoader(target_sr=16000)
denoiser = Denoiser(prop_decrease=0.8)
# 啟用 Wav2vec2 以支援深度特徵
feature_extractor = FeatureExtractor(
    n_mfcc=40,
    n_mels=128,
    use_wav2vec2=True,
    wav2vec2_model_path=settings.agent.voiceprint_wav2vec2_model
)

# 使用 base 模型兼顧展演速度與準確度
transcriber = WhisperTranscriber(model_size="base")

semantic_agent = SemanticAgent(model_path=settings.agent.semantic_model)
semantic_agent.load_model()

voiceprint_agent = VoiceprintAgent(
    prosody_model_path=settings.agent.voiceprint_prosody_model,
    deepfake_model_path=settings.agent.voiceprint_deepfake_model
)
voiceprint_agent.load_model()

memory_agent = MemoryAgent(
    index_path=settings.agent.memory_index_path,
    meta_path=settings.agent.memory_meta_path,
    embedding_model=settings.agent.memory_embedding_model,
)
memory_agent.load_model()

fusion_engine = SEAttentionFusion(model_path=settings.fusion.model_path)
fusion_engine.load_model()

# ========== FastAPI App ==========
app = FastAPI(title="智慧語音詐騙檢測系統", version="3.0.0")

# 靜態檔案
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ========== 非同步記憶庫寫入 worker ==========
memory_write_queue = queue.Queue()

def memory_worker():
    while True:
        try:
            item = memory_write_queue.get()
            if item is None:
                break
                
            case_id = item.get("id")
            text = item.get("text", "")
            fraud_type = item.get("fraud_type", "未知")
            
            # 寫入前檢查字數
            if len(text.strip()) > 5:
                # 取得編碼
                embedding = memory_agent._encode(text)
                memory_agent.persistent_memory.insert(
                    embedding,
                    build_private_case_metadata(case_id, text, fraud_type),
                )
                memory_agent.save()
                logger.info(f"💾 [背景寫入] 成功將通話分析 {case_id[-6:]} 寫入 FAISS 長期防詐庫")
            
            memory_write_queue.task_done()
        except Exception as e:
            logger.error(f"[背景寫入] 發生錯誤: {traceback.format_exc()}")

# 啟動背景執行緒 (deamon=True 隨主程式結束)
worker_thread = threading.Thread(target=memory_worker, daemon=True)
worker_thread.start()

@app.get("/")
async def index():
    """首頁"""
    html_path = STATIC_DIR / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
        
    # 動態注入時間戳以破解 JS/CSS 快取
    t = int(time.time())
    html_content = html_content.replace('app.js?v=3.1', f'app.js?t={t}')
    html_content = html_content.replace('app.js', f'app.js?t={t}')
    html_content = html_content.replace('style.css', f'style.css?t={t}')
    
    return HTMLResponse(
        content=html_content,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
    )


@app.post("/api/analyze")
async def analyze_audio(
    audio: UploadFile = File(...),
    language: str = Form(""),
):
    """核心分析 API

    接收音頻檔案，執行 Step 1~5 分析流程並回傳 JSON 結果。
    """
    start_time = time.time()
    tmp_path = None

    try:
        # === 儲存上傳檔案 ===
        suffix = safe_upload_suffix(audio.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"收到音頻: suffix={suffix}, size={len(content)/1024:.1f}KB")

        # === Step 1: 預處理 ===
        audio_raw, sr = audio_loader.load(tmp_path)
        audio_clean = denoiser.denoise(audio_raw, sr)
        snr = denoiser.estimate_snr(audio_raw, sr)
        features = feature_extractor.extract_all(audio_clean, sr)
        features.snr_estimate = snr
        duration = len(audio_raw) / sr

        # === Step 2: 語音轉錄 (A/B角色) ===
        logger.info("開始 Whisper 語音轉錄...")
        # 若使用者未特別指定語言，預設視為中文 (zh) 提速
        transcript = transcriber.transcribe(
            audio_clean, 
            sr, 
            language=language or "zh"
        )
        transcript_text = transcript.text
        logger.info(f"轉錄完成: chars={len(transcript_text)}")

        # === Step 3: 三 Agent 分析 ===
        voiceprint = voiceprint_agent.analyze(features)
        semantic = semantic_agent.analyze(transcript)
        memory = memory_agent.analyze(transcript)
        agent_results = [voiceprint, semantic, memory]

        # === Step 4: 融合 ===
        fusion = fusion_engine.fuse(agent_results)

        # === Step 5: 組裝回應 ===
        elapsed = time.time() - start_time

        # 韻律特徵
        p = features.prosody
        prosody = {
            "Jitter": p.jitter,
            "Shimmer": p.shimmer,
            "HNR (dB)": round(p.hnr, 2),
            "F0 均值 (Hz)": round(p.f0_mean, 1),
            "F0 標準差": round(p.f0_std, 1),
            "F0 範圍": round(p.f0_range, 1),
            "語速": round(p.speaking_rate, 1),
            "停頓次數": len(p.pause_durations),
        }

        logger.info(
            f"分析完成: {fusion.risk_level}, "
            f"P={fusion.final_probability:.3f}, 耗時 {elapsed:.2f}s"
        )
        
        # 配發一個專屬 Case ID
        current_case_id = str(uuid.uuid4())
        
        # 將結果自動塞入背景寫入隊列 (自動變成長期記憶)
        memory_write_queue.put({
            "id": current_case_id,
            "text": transcript_text,
            "fraud_type": f"{fusion.risk_level}風險 ({fusion.final_probability:.1%})"
        })

        return JSONResponse({
            "id": current_case_id,
            "fraud_probability": fusion.final_probability,
            "risk_level": fusion.risk_level,
            "duration": duration,
            "snr": snr,
            "elapsed": elapsed,
            "transcript": transcript_text,
            "agents": [
                {
                    "name": r.agent_name,
                    "fraud_probability": r.fraud_probability,
                    "confidence": r.confidence,
                    "signal_quality": r.signal_quality,
                    "explanation": r.explanation,
                }
                for r in agent_results
            ],
            "weights": fusion.dynamic_weights,
            "prosody": prosody,
        })

    except Exception as e:
        logger.error(f"分析失敗: {traceback.format_exc()}")
        return JSONResponse(
            {"detail": str(e)},
            status_code=500,
        )
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


@app.get("/api/history")
async def get_history():
    """取得所有長期記憶庫中的案例紀錄"""
    try:
        # 直接讀取 PersistentMemory 中的 metadata，反轉確保最新在上
        metadata = summarize_history_metadata(
            reversed(memory_agent.persistent_memory._metadata)
        )
        return JSONResponse({"history": metadata})
    except Exception as e:
        logger.error(f"獲取歷史紀錄失敗: {e}")
        return JSONResponse({"detail": str(e)}, status_code=500)


@app.delete("/api/history/{case_id}")
async def delete_history(case_id: str):
    """手動刪除長期記憶庫中的特定案例"""
    try:
        success = memory_agent.delete_case(case_id)
        if success:
            logger.info(f"使用者已手動刪除記憶庫案例: {case_id}")
            return JSONResponse({"success": True})
        else:
            return JSONResponse({"success": False, "detail": "找不到該案例 ID"}, status_code=404)
    except Exception as e:
        logger.error(f"刪除歷史紀錄失敗: {e}")
        return JSONResponse({"detail": str(e)}, status_code=500)



# ========== 啟動入口 ==========
if __name__ == "__main__":
    print("═" * 50)
    print("  智慧語音詐騙檢測系統 v3.0")
    print("  http://localhost:7860")
    print("═" * 50)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
