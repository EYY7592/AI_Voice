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
config/                 Runtime settings
src/                    Application source code
static/                 Web UI assets
tests/                  Pytest test suite
notebooks/              Training and experiment notebooks
models/                 Local model artifacts, ignored by git
data/                   Local datasets, ignored by git
logs/                   Runtime logs, ignored by git
```

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

## Tests

```powershell
$env:VIRTUAL_ENV=(Resolve-Path .venv).Path
$env:PATH="$env:VIRTUAL_ENV\Scripts;$env:PATH"
uv run --active pytest tests\test_privacy_controls.py tests\test_step2_step3.py -q
```

## Privacy And Repository Hygiene

Do not commit runtime data, model artifacts, uploaded audio, API keys, tokens, credentials, logs, or raw transcript metadata. The `.gitignore` excludes common sensitive files including:

- `.env`, `.env.*`, `.envrc`
- `*token*`, `*secret*`, `*credential*`, `*api_key*`, `*api-key*`, `*apikey*`
- `client_secret*.json`, `service-account*.json`, `service_account*.json`, `kaggle.json`
- `*.pem`, `*.key`, `*.p12`, `*.pfx`
- `metadata.json`, `data/`, `models/`, `logs/`, audio files, FAISS indexes, model archives

Before publishing, run a secret scan and verify that no raw user transcript or sensitive dataset has been staged.
