# AI_Voice

智慧語音詐騙檢測系統。系統接收通話音訊後，透過音訊預處理、Whisper 語音轉錄、聲紋/語義/記憶三 Agent 分析，以及 SE-Attention 融合，輸出詐騙風險評分與分析結果。

## Features

- FastAPI Web UI 與 REST API
- 音訊預處理、降噪與聲學特徵萃取
- Whisper 中文語音轉錄
- 聲紋異常、語義詐騙模式、歷史記憶比對三路分析
- SE-Attention 融合風險分數
- 隱私保護：長期歷史 API 不回傳原始轉錄文字，log 不保存轉錄片段

## Project Structure

```text
AI_Voice/
├── .env.example                 # 本機環境變數範本，不包含真實密鑰
├── .gitignore                   # 排除 secrets、logs、datasets、audio、model artifacts
├── README.md                    # 專案說明、架構流程與啟動方式
├── requirements.txt             # Python 依賴清單
├── config/
│   └── settings.py              # 全域設定、模型路徑與 runtime 參數
├── src/
│   ├── gui.py                   # FastAPI Web server、REST API、靜態檔案掛載
│   ├── main.py                  # CLI / orchestrator 形式的端到端偵測流程
│   ├── models.py                # 音訊特徵、轉錄、Agent 結果、融合結果 dataclasses
│   ├── privacy.py               # 轉錄隱私保護、metadata 最小化、上傳副檔名清理
│   ├── step1_preprocessing/
│   │   ├── audio_loader.py      # 音訊載入與取樣率正規化
│   │   ├── denoiser.py          # 降噪與 SNR 估算
│   │   └── feature_extractor.py # MFCC、mel、韻律、wav2vec2 特徵萃取
│   ├── step2_transcription/
│   │   ├── whisper_transcriber.py # Whisper 語音轉文字
│   │   └── text_converter.py      # 文字清理與繁體中文正規化
│   ├── step3_agents/
│   │   ├── base_agent.py        # Agent 共用結果封裝
│   │   ├── voiceprint_agent.py  # 聲紋與深偽語音風險分析
│   │   ├── semantic_agent.py    # BERT 詐騙語義分析
│   │   └── memory_agent.py      # Sentence-BERT + FAISS 記憶比對
│   ├── step4_fusion/
│   │   └── se_attention_fusion.py # SE-Attention 動態權重融合
│   ├── step5_report/
│   │   └── report_generator.py  # 分析報告產生支援
│   └── utils/
│       ├── exceptions.py        # 共用例外型別
│       └── logger.py            # 共用 logger 設定
├── static/
│   ├── index.html               # Web UI HTML
│   ├── app.js                   # 上傳、進度、結果、歷史紀錄互動
│   └── style.css                # Web UI 樣式
└── notebooks/
    ├── 01_bert_fraud_training.ipynb        # 語義模型訓練流程
    ├── 02_voiceprint_training.ipynb        # 聲紋模型訓練流程
    └── 03_memory_and_fusion_training.ipynb # 記憶庫與融合模型訓練流程

本地忽略路徑:
├── models/                       # 模型 artifacts 與 FAISS index
├── data/                         # 本地資料集
├── logs/                         # runtime logs
├── metadata.json                 # 原始本地訓練 metadata
└── uploaded audio files           # WAV / MP3 / M4A / FLAC / OGG
```

## Architecture Flow

```mermaid
flowchart TD
    U["使用者在 Web UI 上傳通話音訊"] --> UI["前端介面<br/>static/index.html + static/app.js"]
    UI --> API["FastAPI 分析端點<br/>POST /api/analyze<br/>src/gui.py"]

    API --> TMP["暫存音訊檔<br/>finally 區塊確保清理"]
    TMP --> LDR["音訊載入<br/>AudioLoader<br/>讀取並正規化取樣率"]
    LDR --> DENOISE["音訊降噪<br/>Denoiser<br/>降低背景噪音並估算 SNR"]
    DENOISE --> FEAT["聲學特徵萃取<br/>FeatureExtractor<br/>MFCC / mel / 韻律 / wav2vec2"]

    DENOISE --> ASR["語音轉文字<br/>WhisperTranscriber"]
    ASR --> TXT["文字整理<br/>TextConverter<br/>清理文字並轉為台灣繁體"]

    FEAT --> VAGENT["聲紋 Agent<br/>VoiceprintAgent<br/>韻律異常與深偽語音訊號"]
    TXT --> SAGENT["語義 Agent<br/>SemanticAgent<br/>BERT 詐騙語義分類"]
    TXT --> MAGENT["記憶 Agent<br/>MemoryAgent<br/>Sentence-BERT + FAISS 相似案例比對"]

    VAGENT --> FUSION["融合判決<br/>SEAttentionFusion<br/>動態權重整合三個 Agent"]
    SAGENT --> FUSION
    MAGENT --> FUSION

    FUSION --> RESPONSE["API 回應<br/>風險等級、詐騙機率、Agent 細節、當次轉錄"]
    RESPONSE --> UI

    TXT --> PRIVACY["隱私保護<br/>privacy.py<br/>長期 metadata 只存 hash 與文字長度"]
    PRIVACY --> QUEUE["背景寫入佇列<br/>避免阻塞分析 API"]
    QUEUE --> MEMORY["本地長期記憶庫<br/>models/memory<br/>FAISS index + 去識別 metadata<br/>不進 Git"]

    MEMORY --> HISTORY["歷史紀錄 API<br/>GET /api/history<br/>只回傳遮罩後摘要"]
    HISTORY --> UI
```

## Architecture Details

### Web And API Layer

`static/index.html`, `static/app.js`, and `static/style.css` provide the browser UI. The UI lets a user upload an audio file, shows step-by-step progress, renders the risk score, displays agent-level details, and loads the long-term history view.

`src/gui.py` is the FastAPI application. It mounts the static UI, exposes `POST /api/analyze` for audio analysis, exposes `GET /api/history` for redacted long-term history, and exposes `DELETE /api/history/{case_id}` for removing a memory item. It also owns the background queue that writes analyzed cases into the memory store.

### Step 1: Audio Preprocessing

`src/step1_preprocessing/audio_loader.py` loads the uploaded audio and normalizes it to the configured target sample rate. This gives the downstream models a consistent waveform format.

`src/step1_preprocessing/denoiser.py` reduces background noise and estimates SNR. SNR is returned to the UI as a quality indicator and is also attached to the extracted feature object.

`src/step1_preprocessing/feature_extractor.py` extracts acoustic features such as MFCCs, mel features, prosody, and optional wav2vec2 embeddings. These features feed the voiceprint analysis path.

### Step 2: Transcription

`src/step2_transcription/whisper_transcriber.py` runs Whisper transcription and returns a structured transcript object with text, timing segments, language, and confidence information.

`src/step2_transcription/text_converter.py` cleans transcript text and normalizes Chinese text for downstream semantic analysis. This keeps the BERT and memory matching paths working with a more consistent text format.

### Step 3: Multi-Agent Analysis

`src/step3_agents/voiceprint_agent.py` analyzes acoustic features for signs of synthetic or abnormal voice patterns. It combines prosody and deepfake-model signals into a voiceprint fraud probability.

`src/step3_agents/semantic_agent.py` analyzes the transcript text with the local fraud BERT model. It estimates fraud probability from language patterns and fraud category cues.

`src/step3_agents/memory_agent.py` compares the current transcript embedding against historical cases using Sentence-BERT and FAISS. It returns similarity-based fraud evidence without exposing raw historical transcript text through the history API.

### Step 4: Fusion

`src/step4_fusion/se_attention_fusion.py` combines the three agent outputs. The SE-Attention fusion layer dynamically weights each agent's probability, confidence, and signal quality, then returns the final fraud probability and risk level.

### Privacy Boundary

`src/privacy.py` contains the privacy controls. The current analysis response can return the transcript to the active caller, but persistent memory metadata stores only derived information such as text length and SHA-256 fingerprint. The history API returns redacted placeholders rather than raw transcript content.

Runtime artifacts such as `models/`, `data/`, `logs/`, raw `metadata.json`, audio uploads, FAISS indexes, and model archives are local-only and ignored by git.

## Setup

Use `uv` for Python environment management.

```powershell
uv python install 3.11.15
$env:VIRTUAL_ENV=(Resolve-Path .venv).Path
$env:PATH="$env:VIRTUAL_ENV\Scripts;$env:PATH"
uv run --active python -m uvicorn src.gui:app --host 127.0.0.1 --port 7861 --loop asyncio --http h11 --access-log
```

Open:

```text
http://127.0.0.1:7861/
```

## Privacy And Repository Hygiene

Do not commit runtime data, model artifacts, uploaded audio, API keys, tokens, credentials, logs, or raw transcript metadata. The `.gitignore` excludes common sensitive files including:

- `.env`, `.env.*`, `.envrc`
- `*token*`, `*secret*`, `*credential*`, `*api_key*`, `*api-key*`, `*apikey*`
- `client_secret*.json`, `service-account*.json`, `service_account*.json`, `kaggle.json`
- `*.pem`, `*.key`, `*.p12`, `*.pfx`
- `metadata.json`, `data/`, `models/`, `logs/`, audio files, FAISS indexes, model archives

Before publishing, run a secret scan and verify that no raw user transcript or sensitive dataset has been staged.
