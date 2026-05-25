"""
AI_Voice 資料模型定義
=====================
所有模組之間傳遞的資料結構（dataclass）。
這是系統的「資料契約」，所有模組都依賴這些定義。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# 嘗試匯入 torch，若未安裝則以 None 代替
try:
    import torch
except ImportError:
    torch = None  # type: ignore


# ============================================================
# Step1 輸出：音頻特徵
# ============================================================

@dataclass
class ProsodyFeatures:
    """韻律特徵（由 parselmouth/Praat 萃取）

    用於偵測 AI 合成語音的微觀聲學異常。
    AI 語音在這些特徵上通常呈現「過於完美」的模式。
    """
    jitter: float = 0.0              # 基頻微抖動（%），AI 語音通常 < 0.5%
    shimmer: float = 0.0             # 振幅微抖動（%），AI 語音變異過低
    hnr: float = 0.0                 # 諧波噪聲比（dB），合成語音異常高
    f0_mean: float = 0.0             # 基頻均值（Hz）
    f0_std: float = 0.0              # 基頻標準差（Hz）
    f0_range: float = 0.0            # 基頻範圍（Hz）
    speaking_rate: float = 0.0       # 語速（音節/秒）
    pause_durations: list[float] = field(default_factory=list)  # 停頓時長清單（秒）
    formants: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])  # [F1,F2,F3,F4]（Hz）

    def to_feature_vector(self) -> np.ndarray:
        """將韻律特徵轉換為固定長度的特徵向量（供模型輸入）

        Returns:
            np.ndarray: 形狀為 (20,) 的特徵向量
        """
        # 停頓統計：均值、標準差、最大值、計數
        pause_arr = np.array(self.pause_durations) if self.pause_durations else np.array([0.0])
        pause_mean = float(np.mean(pause_arr))
        pause_std = float(np.std(pause_arr))
        pause_max = float(np.max(pause_arr))
        pause_count = len(self.pause_durations)

        return np.array([
            self.jitter, self.shimmer, self.hnr,
            self.f0_mean, self.f0_std, self.f0_range,
            self.speaking_rate,
            pause_mean, pause_std, pause_max, pause_count,
            *self.formants[:4],  # F1~F4
            # 衍生特徵（5 個）
            self.f0_std / max(self.f0_mean, 1e-6),        # F0 變異係數
            self.shimmer / max(self.jitter, 1e-6),         # Shimmer/Jitter 比值
            self.hnr / max(self.f0_std, 1e-6),             # HNR 正規化
            pause_std / max(pause_mean, 1e-6),             # 停頓變異係數
            self.jitter + self.shimmer,                     # Jitter+Shimmer 總不穩定度
        ], dtype=np.float32)


@dataclass
class AudioFeatures:
    """音頻特徵集合（Step1 的完整輸出）

    包含三大類特徵：通用聲學 + 韻律 + 深度特徵。
    """
    mfcc: np.ndarray = field(default_factory=lambda: np.zeros((40, 1)))
    mel_spectrogram: np.ndarray = field(default_factory=lambda: np.zeros((128, 1)))
    zcr: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    spectral_centroid: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    prosody: ProsodyFeatures = field(default_factory=ProsodyFeatures)
    wav2vec2_features: Any = None  # torch.Tensor | None（避免強制依賴 torch）
    duration: float = 0.0         # 音頻時長（秒）
    snr_estimate: float = 0.0     # 估計信噪比（dB）


# ============================================================
# Step2 輸出：轉錄結果
# ============================================================

@dataclass
class TranscriptionSegment:
    """轉錄片段（帶時間戳）"""
    start: float = 0.0    # 開始時間（秒）
    end: float = 0.0      # 結束時間（秒）
    text: str = ""         # 片段文字


@dataclass
class TranscriptionResult:
    """轉錄結果（Step2 的完整輸出）"""
    text: str = ""                                       # 完整文字稿（繁體中文）
    segments: list[TranscriptionSegment] = field(default_factory=list)  # 帶時間戳片段
    language: str = "zh"                                 # 偵測到的語言代碼
    confidence: float = 0.0                              # 轉錄信心度 [0, 1]

    @property
    def word_count(self) -> int:
        """文字稿字數"""
        return len(self.text.replace(" ", ""))

    @property
    def total_duration(self) -> float:
        """音頻總時長（秒），基於片段時間戳"""
        if not self.segments:
            return 0.0
        return self.segments[-1].end - self.segments[0].start


# ============================================================
# Step3 輸出：Agent 分析結果
# ============================================================

@dataclass
class AgentResult:
    """Agent 分析結果（所有 Agent 的統一輸出格式）

    Attributes:
        agent_name: Agent 名稱（voiceprint / semantic / memory）
        fraud_probability: 詐騙機率 [0, 1]
        confidence: 模型信心度 [0, 1]
        signal_quality: 信號品質 [0, 1]（供動態權重使用）
        details: 詳細分析資訊（Agent 特定的額外資料）
        explanation: 人類可讀的分析說明
    """
    agent_name: str = ""
    fraud_probability: float = 0.0
    confidence: float = 0.0
    signal_quality: float = 0.0
    details: dict = field(default_factory=dict)
    explanation: str = ""

    def __post_init__(self):
        """驗證數值範圍"""
        self.fraud_probability = max(0.0, min(1.0, self.fraud_probability))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.signal_quality = max(0.0, min(1.0, self.signal_quality))


# ============================================================
# Step3-記憶系統：記憶匹配結果
# ============================================================

@dataclass
class MemoryMatch:
    """記憶匹配結果"""
    similarity: float = 0.0    # 餘弦相似度 [0, 1]
    case_text: str = ""         # 匹配案例文字
    fraud_type: str = ""        # 詐騙類型
    timestamp: str = ""         # 案例時間戳


# ============================================================
# Step4 輸出：融合判決結果
# ============================================================

@dataclass
class FusionResult:
    """融合判決結果（系統最終輸出）"""
    final_probability: float = 0.0                       # 最終詐騙機率 [0, 1]
    risk_level: str = "低風險"                            # "高風險" / "中風險" / "低風險"
    dynamic_weights: dict[str, float] = field(default_factory=dict)  # 各 Agent 動態權重
    agent_results: list[AgentResult] = field(default_factory=list)   # 原始 Agent 結果

    def __post_init__(self):
        """自動判定風險等級"""
        if self.final_probability >= 0.7:
            self.risk_level = "高風險"
        elif self.final_probability >= 0.4:
            self.risk_level = "中風險"
        else:
            self.risk_level = "低風險"
