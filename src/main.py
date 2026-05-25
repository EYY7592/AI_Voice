"""
AI_Voice 核心協調器 (Orchestrator)
==================================
負責串接 Step1 到 Step5 的完整工作流。
"""
import time
from pathlib import Path
from typing import Optional

from src.step1_preprocessing.audio_loader import AudioLoader
from src.step1_preprocessing.denoiser import Denoiser
from src.step1_preprocessing.feature_extractor import FeatureExtractor
from src.step2_transcription.whisper_transcriber import WhisperTranscriber
from src.step2_transcription.text_converter import TextConverter
from src.step3_agents.voiceprint_agent import VoiceprintAgent
from src.step3_agents.semantic_agent import SemanticAgent
from src.step3_agents.memory_agent import MemoryAgent
from src.step4_fusion.se_attention_fusion import SEAttentionFusion
from src.step5_report.report_generator import ReportGenerator
from src.utils.logger import setup_logger
from config.settings import settings

logger = setup_logger("ai_voice.core", level="INFO")

class AIVoiceDetector:
    """AI_Voice 檢測系統核心類別"""

    def __init__(self, model_root: str = "models") -> None:
        logger.info("正在初始化 AI_Voice 檢測系統...")
        
        # Step 1
        self.loader = AudioLoader()
        self.denoiser = Denoiser()
        self.extractor = FeatureExtractor(
            use_wav2vec2=True,
            wav2vec2_model_path=settings.agent.voiceprint_wav2vec2_model
        )
        
        # Step 2
        self.whisper = WhisperTranscriber(model_size="base") # 預設使用 base 兼顧速度
        self.converter = TextConverter()
        
        # Step 3
        self.agent_v = VoiceprintAgent(
            prosody_model_path=settings.agent.voiceprint_prosody_model,
            deepfake_model_path=settings.agent.voiceprint_deepfake_model
        )
        self.agent_s = SemanticAgent(model_path=settings.agent.semantic_model)
        self.agent_m = MemoryAgent(
            index_path=settings.agent.memory_index_path,
            meta_path=settings.agent.memory_meta_path,
            embedding_model=settings.agent.memory_embedding_model
        )
        
        # Step 4
        self.fusion_engine = SEAttentionFusion(model_path=settings.fusion.model_path)
        
        # Step 5
        self.reporter = ReportGenerator()
        
        # 載入所有模型
        self._load_all_models()

    def _load_all_models(self):
        """一次性載入所有 Agent 和融合模型"""
        self.agent_v.load_model()
        self.agent_s.load_model()
        self.agent_m.load_model()
        self.fusion_engine.load_model()

    def detect(self, audio_path: str) -> str:
        """執行完整檢測流程

        Args:
            audio_path: 音頻檔案路徑

        Returns:
            生成的報告檔案路徑
        """
        start_time = time.time()
        logger.info(f"開始分析任務: {audio_path}")
        
        # 1. 預處理
        raw_audio, sr = self.loader.load(audio_path)
        clean_audio = self.denoiser.denoise(raw_audio, sr)
        snr_val = self.denoiser.estimate_snr(raw_audio, sr)
        
        features = self.extractor.extract_all(clean_audio, sr)
        features.snr_estimate = snr_val
        
        # 2. 轉錄
        raw_transcript = self.whisper.transcribe(clean_audio, sr)
        # 繁體化與清理
        raw_transcript.text = self.converter.process(raw_transcript.text)
        
        # 3. Agent 檢測
        res_v = self.agent_v.analyze(features)
        res_s = self.agent_s.analyze(raw_transcript)
        res_m = self.agent_m.analyze(raw_transcript)
        
        # 4. 融合
        fusion_result = self.fusion_engine.fuse([res_v, res_s, res_m])
        
        # 5. 報告
        elapsed = time.time() - start_time
        report_path = self.reporter.generate(fusion_result, features, elapsed)
        
        logger.info(f"任務完成！風險等級: {fusion_result.risk_level}, 耗時: {elapsed:.2f}s")
        return report_path

if __name__ == "__main__":
    # 簡單 CLI 測試
    import sys
    if len(sys.argv) > 1:
        detector = AIVoiceDetector()
        report = detector.detect(sys.argv[1])
        print(f"分析報告已儲存於: {report}")
