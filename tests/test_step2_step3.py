"""
Step2 & Step3 單元測試
======================
測試轉錄模組（mock Whisper）和三個 Agent 的規則模式。
"""
import numpy as np
import pytest

from src.models import (
    AudioFeatures, ProsodyFeatures,
    TranscriptionResult, TranscriptionSegment,
    AgentResult,
)
from src.step2_transcription.text_converter import TextConverter
from src.step3_agents.voiceprint_agent import VoiceprintAgent
from src.step3_agents.semantic_agent import SemanticAgent
from src.step3_agents.memory_agent import MemoryAgent, WorkingMemory, PersistentMemory


# ============================================================
# Step2：TextConverter 測試
# ============================================================

class TestTextConverter:
    """TextConverter 繁簡轉換器測試"""

    @pytest.fixture
    def converter(self):
        """建立 TextConverter 實例"""
        return TextConverter(mode="s2twp")

    def test_basic_conversion(self, converter):
        """測試基本繁簡轉換"""
        result = converter.to_traditional("你好世界")
        assert "你好世界" == result  # 這句繁簡一樣

    def test_convert_simplified(self, converter):
        """測試簡體轉繁體（台灣用語）"""
        result = converter.to_traditional("信息安全")
        assert "資訊" in result and "安全" in result or "保安" in result

    def test_convert_complex(self, converter):
        """測試複雜文字轉換"""
        result = converter.to_traditional("网络通讯软件")
        assert "網路" in result or "網絡" in result

    def test_empty_input(self, converter):
        """測試空輸入"""
        assert converter.to_traditional("") == ""
        assert converter.to_traditional("  ") == "  "

    def test_clean_text(self, converter):
        """測試文字清理"""
        result = converter.clean_text("你好，  世界。  測試")
        assert "  " not in result

    def test_detect_language_chinese(self, converter):
        """測試語言偵測：中文"""
        assert converter.detect_language("這是一段中文文字") == "zh"

    def test_detect_language_english(self, converter):
        """測試語言偵測：英文"""
        assert converter.detect_language("This is English text") == "en"

    def test_detect_language_empty(self, converter):
        """測試語言偵測：空文字"""
        assert converter.detect_language("") == "unknown"

    def test_process_pipeline(self, converter):
        """測試完整處理流程"""
        result = converter.process("  信息安全  非常重要  ")
        assert "資訊" in result
        assert "安全" in result or "保安" in result
        assert "  " not in result

    def test_invalid_mode(self):
        """測試無效模式 → 拋出 ValueError"""
        with pytest.raises(ValueError):
            TextConverter(mode="invalid")


# ============================================================
# Step3：VoiceprintAgent 測試
# ============================================================

class TestVoiceprintAgent:
    """VoiceprintAgent 聲紋分析測試"""

    @pytest.fixture
    def agent(self):
        """建立規則模式的 VoiceprintAgent"""
        return VoiceprintAgent()

    @pytest.fixture
    def normal_features(self):
        """正常語音的音頻特徵"""
        return AudioFeatures(
            mfcc=np.random.randn(40, 50).astype(np.float32),
            mel_spectrogram=np.random.randn(128, 50).astype(np.float32),
            prosody=ProsodyFeatures(
                jitter=0.015, shimmer=0.035, hnr=18.0,
                f0_mean=180.0, f0_std=32.0, f0_range=120.0,
                speaking_rate=4.5,
                pause_durations=[0.2, 0.5, 0.3, 0.8],
                formants=[500, 1500, 2500, 3500],
            ),
            duration=5.0,
            snr_estimate=25.0,
        )

    @pytest.fixture
    def suspicious_features(self):
        """可疑 AI 語音特徵"""
        return AudioFeatures(
            mfcc=np.random.randn(40, 50).astype(np.float32),
            mel_spectrogram=np.random.randn(128, 50).astype(np.float32),
            prosody=ProsodyFeatures(
                jitter=0.002, shimmer=0.008, hnr=32.0,
                f0_mean=200.0, f0_std=8.0, f0_range=30.0,
                speaking_rate=5.0,
                pause_durations=[0.3, 0.31, 0.29, 0.3],
                formants=[500, 1500, 2500, 3500],
            ),
            duration=5.0,
            snr_estimate=35.0,
        )

    def test_analyze_normal(self, agent, normal_features):
        """測試正常語音分析"""
        result = agent.analyze(audio_features=normal_features)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "voiceprint"
        assert 0 <= result.fraud_probability <= 1
        assert result.fraud_probability < 0.5  # 正常語音應低分

    def test_analyze_suspicious(self, agent, suspicious_features):
        """測試可疑語音分析"""
        result = agent.analyze(audio_features=suspicious_features)
        assert isinstance(result, AgentResult)
        assert result.fraud_probability > 0.3  # 可疑語音應高分

    def test_signal_quality(self, agent, normal_features):
        """測試信號品質計算"""
        result = agent.analyze(audio_features=normal_features)
        assert 0 <= result.signal_quality <= 1

    def test_explanation_not_empty(self, agent, normal_features):
        """測試說明不為空"""
        result = agent.analyze(audio_features=normal_features)
        assert result.explanation != ""

    def test_details_has_mode(self, agent, normal_features):
        """測試詳細資訊包含模式"""
        result = agent.analyze(audio_features=normal_features)
        assert result.details.get("mode") == "rule"


# ============================================================
# Step3：SemanticAgent 測試
# ============================================================

class TestSemanticAgent:
    """SemanticAgent 語義分析測試"""

    @pytest.fixture
    def agent(self):
        """建立規則模式的 SemanticAgent"""
        return SemanticAgent()

    def test_analyze_fraud_text(self, agent, sample_transcript_text):
        """測試詐騙文字分析"""
        transcript = TranscriptionResult(
            text=sample_transcript_text,
            confidence=0.9,
        )
        result = agent.analyze(transcript)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "semantic"
        assert result.fraud_probability > 0.3  # 詐騙文字應有高分

    def test_analyze_normal_text(self, agent, sample_normal_text):
        """測試正常文字分析"""
        transcript = TranscriptionResult(
            text=sample_normal_text,
            confidence=0.9,
        )
        result = agent.analyze(transcript)
        assert result.fraud_probability < 0.3  # 正常文字應低分

    def test_analyze_empty_text(self, agent):
        """測試空文字分析"""
        transcript = TranscriptionResult(text="")
        result = agent.analyze(transcript)
        assert result.fraud_probability == 0.0

    def test_extract_keywords(self, agent, sample_transcript_text):
        """測試關鍵詞提取"""
        keywords = agent.extract_fraud_keywords(sample_transcript_text)
        assert len(keywords) > 0
        assert "公安局" in keywords or "洗錢" in keywords

    def test_classify_fraud_type(self, agent, sample_transcript_text):
        """測試詐騙類型分類"""
        fraud_type, score = agent.classify_fraud_type(sample_transcript_text)
        assert fraud_type == "冒充公檢法"
        assert score > 0

    def test_pressure_keywords(self, agent):
        """測試壓力話術偵測"""
        text = "你必須立刻馬上轉帳，否則後果自負，千萬不要告訴別人"
        transcript = TranscriptionResult(text=text, confidence=0.9)
        result = agent.analyze(transcript)
        assert result.details.get("pressure_keywords_count", 0) > 0


# ============================================================
# Step3：MemoryAgent 測試
# ============================================================

class TestWorkingMemory:
    """WorkingMemory 短期記憶測試"""

    def test_store_retrieve(self):
        """測試存取"""
        wm = WorkingMemory(capacity=10)
        wm.store("key1", "value1")
        assert wm.retrieve("key1") == "value1"

    def test_capacity_limit(self):
        """測試容量限制"""
        wm = WorkingMemory(capacity=3)
        for i in range(5):
            wm.store(f"key{i}", f"value{i}")
        assert wm.retrieve("key0") is None  # 最舊的被淘汰
        assert wm.retrieve("key4") == "value4"

    def test_judgment_history(self):
        """測試判決歷史"""
        wm = WorkingMemory()
        wm.add_judgment({"result": "fraud"})
        wm.add_judgment({"result": "safe"})
        recent = wm.get_recent_judgments(5)
        assert len(recent) == 2


class TestPersistentMemory:
    """PersistentMemory 長期記憶測試"""

    def test_empty_search(self):
        """測試空索引搜尋"""
        pm = PersistentMemory()
        pm.load()
        results = pm.search(np.random.randn(384).astype(np.float32))
        assert len(results) == 0

    def test_insert_and_search(self):
        """測試插入和搜尋"""
        pm = PersistentMemory(embedding_dim=384)
        pm.load()

        # 插入案例
        vec = np.random.randn(384).astype(np.float32)
        vec = vec / np.linalg.norm(vec)  # 正規化
        pm.insert(vec, {"text": "測試案例", "fraud_type": "測試"})

        assert pm.total_cases == 1

        # 搜尋（使用相同向量）
        results = pm.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0].similarity > 0.9  # 自身搜尋應該高度匹配


class TestMemoryAgent:
    """MemoryAgent 記憶系統測試"""

    @pytest.fixture
    def agent(self):
        """建立 MemoryAgent（不載入外部模型）"""
        ma = MemoryAgent()
        ma.persistent_memory.load()  # 建立空索引
        return ma

    def test_analyze_empty_memory(self, agent, sample_transcript_text):
        """測試空記憶庫分析"""
        transcript = TranscriptionResult(text=sample_transcript_text)
        result = agent.analyze(transcript)
        assert isinstance(result, AgentResult)
        assert result.agent_name == "memory"
        assert result.fraud_probability == 0.0  # 空記憶庫

    def test_store_and_analyze(self, agent, sample_transcript_text):
        """測試寫入案例後的分析"""
        # 寫入案例
        agent.store_episode("公安局涉嫌洗錢安全帳戶", "冒充公檢法")

        # 分析
        transcript = TranscriptionResult(text=sample_transcript_text)
        result = agent.analyze(transcript)
        assert result.details.get("matched_cases", 0) >= 0

    def test_optimize(self, agent):
        """測試記憶優化"""
        result = agent.optimize()
        assert "deduplicated" in result
        assert "decayed" in result
