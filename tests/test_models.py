"""
資料模型（dataclass）單元測試
==============================
驗證所有資料結構的建立、驗證邏輯和序列化方法。
"""
import numpy as np
import pytest

from src.models import (
    ProsodyFeatures,
    AudioFeatures,
    TranscriptionSegment,
    TranscriptionResult,
    AgentResult,
    MemoryMatch,
    FusionResult,
)


class TestProsodyFeatures:
    """ProsodyFeatures 韻律特徵測試"""

    def test_default_creation(self):
        """測試預設值建立"""
        pf = ProsodyFeatures()
        assert pf.jitter == 0.0
        assert pf.shimmer == 0.0
        assert pf.hnr == 0.0
        assert len(pf.formants) == 4

    def test_custom_values(self):
        """測試自訂值建立"""
        pf = ProsodyFeatures(
            jitter=0.015, shimmer=0.03, hnr=20.0,
            f0_mean=180.0, f0_std=25.0, f0_range=100.0,
            speaking_rate=4.5,
            pause_durations=[0.2, 0.5, 0.3],
            formants=[500.0, 1500.0, 2500.0, 3500.0],
        )
        assert pf.jitter == 0.015
        assert pf.f0_mean == 180.0
        assert len(pf.pause_durations) == 3

    def test_to_feature_vector_shape(self):
        """測試特徵向量形狀固定為 20 維"""
        pf = ProsodyFeatures(
            jitter=0.01, shimmer=0.03, hnr=15.0,
            f0_mean=200.0, f0_std=30.0, f0_range=120.0,
            speaking_rate=5.0,
            pause_durations=[0.1, 0.3],
            formants=[500, 1500, 2500, 3500],
        )
        vec = pf.to_feature_vector()
        assert vec.shape == (20,)
        assert vec.dtype == np.float32

    def test_to_feature_vector_empty_pauses(self):
        """測試空停頓清單時特徵向量不出錯"""
        pf = ProsodyFeatures()
        vec = pf.to_feature_vector()
        assert vec.shape == (20,)
        assert not np.any(np.isnan(vec))

    def test_to_feature_vector_values(self):
        """測試特徵向量的值是否正確對應"""
        pf = ProsodyFeatures(jitter=0.02, shimmer=0.05, hnr=18.0)
        vec = pf.to_feature_vector()
        assert vec[0] == pytest.approx(0.02)   # jitter
        assert vec[1] == pytest.approx(0.05)   # shimmer
        assert vec[2] == pytest.approx(18.0)   # hnr


class TestAudioFeatures:
    """AudioFeatures 音頻特徵測試"""

    def test_default_creation(self):
        """測試預設值建立"""
        af = AudioFeatures()
        assert af.mfcc.shape == (40, 1)
        assert af.mel_spectrogram.shape == (128, 1)
        assert af.wav2vec2_features is None
        assert af.duration == 0.0

    def test_custom_shapes(self):
        """測試自訂形狀"""
        af = AudioFeatures(
            mfcc=np.zeros((40, 100)),
            mel_spectrogram=np.zeros((128, 100)),
            duration=3.2,
            snr_estimate=25.0,
        )
        assert af.mfcc.shape == (40, 100)
        assert af.duration == 3.2
        assert af.snr_estimate == 25.0


class TestTranscriptionResult:
    """TranscriptionResult 轉錄結果測試"""

    def test_default_creation(self):
        """測試預設值建立"""
        tr = TranscriptionResult()
        assert tr.text == ""
        assert tr.word_count == 0
        assert tr.total_duration == 0.0

    def test_word_count(self):
        """測試字數計算（忽略空格）"""
        tr = TranscriptionResult(text="你好 世界 測試")
        assert tr.word_count == 6  # 不含空格

    def test_total_duration(self):
        """測試總時長計算"""
        tr = TranscriptionResult(
            segments=[
                TranscriptionSegment(start=0.0, end=2.0, text="你好"),
                TranscriptionSegment(start=2.5, end=5.0, text="世界"),
            ]
        )
        assert tr.total_duration == pytest.approx(5.0)


class TestAgentResult:
    """AgentResult 分析結果測試"""

    def test_default_creation(self):
        """測試預設值建立"""
        ar = AgentResult()
        assert ar.fraud_probability == 0.0
        assert ar.confidence == 0.0

    def test_value_clamping(self):
        """測試數值範圍自動裁切"""
        ar = AgentResult(fraud_probability=1.5, confidence=-0.3, signal_quality=2.0)
        assert ar.fraud_probability == 1.0   # 裁切到 1.0
        assert ar.confidence == 0.0          # 裁切到 0.0
        assert ar.signal_quality == 1.0      # 裁切到 1.0

    def test_normal_values(self):
        """測試正常數值不被改變"""
        ar = AgentResult(
            agent_name="voiceprint",
            fraud_probability=0.82,
            confidence=0.90,
            signal_quality=0.95,
        )
        assert ar.fraud_probability == pytest.approx(0.82)
        assert ar.confidence == pytest.approx(0.90)


class TestFusionResult:
    """FusionResult 融合判決結果測試"""

    def test_high_risk(self):
        """測試高風險自動判定"""
        fr = FusionResult(final_probability=0.85)
        assert fr.risk_level == "高風險"

    def test_medium_risk(self):
        """測試中風險自動判定"""
        fr = FusionResult(final_probability=0.55)
        assert fr.risk_level == "中風險"

    def test_low_risk(self):
        """測試低風險自動判定"""
        fr = FusionResult(final_probability=0.20)
        assert fr.risk_level == "低風險"

    def test_boundary_high(self):
        """測試邊界值：剛好 0.7（高風險）"""
        fr = FusionResult(final_probability=0.7)
        assert fr.risk_level == "高風險"

    def test_boundary_medium(self):
        """測試邊界值：剛好 0.4（中風險）"""
        fr = FusionResult(final_probability=0.4)
        assert fr.risk_level == "中風險"

    def test_zero(self):
        """測試零機率"""
        fr = FusionResult(final_probability=0.0)
        assert fr.risk_level == "低風險"


class TestMemoryMatch:
    """MemoryMatch 記憶匹配結果測試"""

    def test_creation(self):
        """測試建立"""
        mm = MemoryMatch(
            similarity=0.93,
            case_text="涉嫌洗錢",
            fraud_type="冒充公檢法",
            timestamp="2026-01-01",
        )
        assert mm.similarity == 0.93
        assert mm.fraud_type == "冒充公檢法"
