"""
AI_Voice 專案全域設定
=====================
智慧語音詐騙檢測工具的所有可調參數集中管理。
"""
import os
from pathlib import Path
from dataclasses import dataclass, field


# === 專案根目錄 ===
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class AudioConfig:
    """音頻處理參數"""
    target_sr: int = 16000                # 目標取樣率（Hz）
    max_duration: float = 300.0           # 最大處理秒數（5分鐘）
    segment_duration: float = 30.0        # 分段長度（秒）
    silence_top_db: int = 30              # 靜音偵測閾值（dB）
    denoise_prop_decrease: float = 0.8    # 降噪強度（0~1）
    n_mfcc: int = 40                      # MFCC 係數數量
    n_mels: int = 128                     # Mel 頻帶數量
    use_wav2vec2: bool = True             # 是否提取 Wav2vec2 深度特徵


@dataclass
class WhisperConfig:
    """Whisper 轉錄參數"""
    model_size: str = "medium"            # 模型大小：tiny/base/small/medium/large-v3
    language: str = "zh"                  # 指定語言（跳過語言偵測）
    device: str = "cuda"                  # 推理裝置
    fp16: bool = True                     # 是否使用半精度推理


@dataclass
class AgentConfig:
    """Agent 相關參數"""
    # 聲紋分析 Agent
    voiceprint_prosody_model: str = str(PROJECT_ROOT / "models" / "voiceprint" / "lgbm_prosody.pkl")
    voiceprint_deepfake_model: str = str(PROJECT_ROOT / "models" / "voiceprint" / "deepfake_cnn.pt")
    voiceprint_wav2vec2_model: str = str(PROJECT_ROOT / "models" / "wav2vec2")
    voiceprint_internal_alpha: float = 0.4   # 韻律子分數權重
    voiceprint_internal_beta: float = 0.6    # 深偽子分數權重 (CNN 更可靠)

    # 語義分析 Agent
    semantic_model: str = str(PROJECT_ROOT / "models" / "bert_fraud")
    semantic_max_length: int = 256           # BERT 最大輸入長度
    semantic_num_classes: int = 11           # 10 類詐騙 + 1 正常

    # 記憶系統 Agent
    memory_index_path: str = str(PROJECT_ROOT / "models" / "memory" / "faiss.index")
    memory_meta_path: str = str(PROJECT_ROOT / "models" / "memory" / "metadata.json")
    memory_embedding_model: str = str(PROJECT_ROOT / "models" / "sentence_bert")
    memory_top_k: int = 5                    # 檢索 Top-K 案例
    memory_dedup_threshold: float = 0.95     # 去重相似度閾值
    memory_decay_rate: float = 0.01          # 時間衰減速率


@dataclass
class FusionConfig:
    """融合判決參數"""
    model_path: str = str(PROJECT_ROOT / "models" / "fusion" / "se_attention_mlp.pt")
    n_agents: int = 3                        # Agent 數量
    features_per_agent: int = 3              # 每 Agent 特徵數 (P, C, Q)
    hidden_dim: int = 16                     # SE-Attention 隱藏層維度
    high_risk_threshold: float = 0.7         # 高風險閾值
    medium_risk_threshold: float = 0.4       # 中風險閾值


@dataclass
class OutputConfig:
    """輸出設定"""
    template_dir: str = str(PROJECT_ROOT / "src" / "step5_output" / "templates")
    default_format: str = "json"             # 預設輸出格式：json / html / console
    report_output_dir: str = str(PROJECT_ROOT / "reports")


@dataclass
class Settings:
    """全域設定彙總"""
    audio: AudioConfig = field(default_factory=AudioConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # 繁簡轉換模式
    opencc_mode: str = "s2twp"               # 簡體→繁體（台灣用語偏好）

    # 日誌設定
    log_level: str = "INFO"
    log_file: str = str(PROJECT_ROOT / "logs" / "ai_voice.log")


# === 全域設定單例 ===
settings = Settings()
