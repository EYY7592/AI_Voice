"""
AI_Voice 自定義例外
===================
專案統一的例外類別層次結構。
"""


class AIVoiceError(Exception):
    """AI_Voice 專案的基底例外類別"""
    pass


# === Step1：音頻處理相關例外 ===

class AudioProcessingError(AIVoiceError):
    """音頻處理過程中的通用錯誤"""
    pass


class AudioLoadError(AudioProcessingError):
    """音頻載入失敗（格式不支援、檔案損壞、檔案不存在等）"""
    pass


class AudioDenoiseError(AudioProcessingError):
    """降噪處理失敗"""
    pass


class FeatureExtractionError(AudioProcessingError):
    """特徵萃取失敗"""
    pass


# === Step2：轉錄相關例外 ===

class TranscriptionError(AIVoiceError):
    """語音轉錄過程中的通用錯誤"""
    pass


class WhisperModelError(TranscriptionError):
    """Whisper 模型載入或推理失敗"""
    pass


class TextConversionError(TranscriptionError):
    """繁簡轉換失敗"""
    pass


# === Step3：Agent 相關例外 ===

class AgentError(AIVoiceError):
    """Agent 分析過程中的通用錯誤"""
    pass


class ModelLoadError(AgentError):
    """模型權重載入失敗"""
    pass


class AgentAnalysisError(AgentError):
    """Agent 分析執行失敗"""
    pass


# === Step4：融合判決相關例外 ===

class FusionError(AIVoiceError):
    """融合判決過程中的錯誤"""
    pass


# === Step5：輸出相關例外 ===

class ReportGenerationError(AIVoiceError):
    """報告生成失敗"""
    pass


# === 記憶系統相關例外 ===

class MemoryError(AIVoiceError):
    """記憶系統操作失敗"""
    pass


class MemoryIndexError(MemoryError):
    """FAISS 索引操作失敗"""
    pass


class MemorySearchError(MemoryError):
    """記憶檢索失敗"""
    pass
