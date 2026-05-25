"""
AI_Voice 測試共用 fixtures
==========================
提供所有測試共用的假資料、mock 物件和工具函式。
"""
import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """生成一段合成測試音頻（1 秒的 440Hz 正弦波 + 噪音）

    Returns:
        tuple: (audio_array, sample_rate)
    """
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440Hz 正弦波（A4 音符）
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    # 加入少量隨機噪音
    audio += 0.05 * np.random.randn(len(audio))
    return audio.astype(np.float32), sr


@pytest.fixture
def empty_audio():
    """生成空白（靜音）音頻

    Returns:
        tuple: (audio_array, sample_rate)
    """
    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)  # 1 秒靜音
    return audio, sr


@pytest.fixture
def short_audio():
    """生成極短音頻（0.05 秒）

    Returns:
        tuple: (audio_array, sample_rate)
    """
    sr = 16000
    duration = 0.05
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def long_audio():
    """生成較長音頻（60 秒，用於測試分段）

    Returns:
        tuple: (audio_array, sample_rate)
    """
    sr = 16000
    duration = 60.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def sample_transcript_text():
    """提供一段模擬的詐騙轉錄文字

    Returns:
        str: 模擬詐騙文字稿
    """
    return (
        "你好，我是公安局的張警官。你的銀行帳戶涉嫌洗錢，"
        "現在需要你立即將資金轉入安全帳戶。請馬上配合，"
        "否則我們將對你採取強制措施。"
    )


@pytest.fixture
def sample_normal_text():
    """提供一段正常對話文字

    Returns:
        str: 正常對話文字稿
    """
    return (
        "嗨，你最近好嗎？週末有空一起吃飯嗎？"
        "我找到一家新開的餐廳，聽說評價不錯。"
    )


@pytest.fixture
def mock_agent_results():
    """生成模擬的三個 Agent 分析結果

    Returns:
        list[dict]: 三個模擬 AgentResult 的字典表示
    """
    return [
        {
            "agent_name": "voiceprint",
            "fraud_probability": 0.82,
            "confidence": 0.90,
            "signal_quality": 0.95,
            "details": {"jitter": 0.003, "shimmer": 0.02, "hnr": 25.0},
            "explanation": "韻律特徵顯示高度合成語音可能性",
        },
        {
            "agent_name": "semantic",
            "fraud_probability": 0.75,
            "confidence": 0.85,
            "signal_quality": 0.80,
            "details": {"fraud_type": "冒充公檢法", "keywords": ["公安局", "洗錢", "安全帳戶"]},
            "explanation": "文字稿包含典型冒充公檢法詐騙模式",
        },
        {
            "agent_name": "memory",
            "fraud_probability": 0.91,
            "confidence": 0.88,
            "signal_quality": 0.92,
            "details": {"top_match_similarity": 0.93, "matched_cases": 3},
            "explanation": "記憶庫中找到 3 個高度相似的歷史詐騙案例",
        },
    ]
