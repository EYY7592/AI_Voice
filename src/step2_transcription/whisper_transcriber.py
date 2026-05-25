"""
Whisper 離線語音轉錄器
======================
使用 OpenAI Whisper 模型進行完全離線的語音轉文字轉錄。
支援多種模型大小、語言偵測和帶時間戳的片段輸出。
"""
import numpy as np

from src.models import TranscriptionResult, TranscriptionSegment
from src.utils.logger import get_logger
from src.utils.exceptions import WhisperModelError

logger = get_logger("ai_voice.step2.whisper")


class WhisperTranscriber:
    """Whisper 離線轉錄器

    將音頻轉錄為帶時間戳的文字稿。模型完全離線運行，
    首次使用時會自動下載模型權重至本機快取。

    Attributes:
        model_size: 模型大小（tiny/base/small/medium/large）
        device: 推理裝置
    """

    # 可用模型及其近似 VRAM 需求
    MODEL_SIZES = {
        "tiny": "~1GB",
        "base": "~1GB",
        "small": "~2GB",
        "medium": "~5GB",
        "large": "~10GB",
    }

    def __init__(
        self,
        model_size: str = "medium",
        device: str | None = None
    ) -> None:
        """初始化

        Args:
            model_size: Whisper 模型大小（tiny/base/small/medium/large）
            device: 推理裝置。None 時自動偵測（有 CUDA 用 CUDA，否則 CPU）
        """
        if model_size not in self.MODEL_SIZES:
            raise WhisperModelError(
                f"不支援的模型大小: {model_size}。"
                f"可用選項: {list(self.MODEL_SIZES.keys())}"
            )

        self.model_size = model_size
        self._model = None  # 延遲載入

        # 自動偵測裝置
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(
            f"WhisperTranscriber 初始化: "
            f"模型={model_size} ({self.MODEL_SIZES[model_size]}), "
            f"裝置={self.device}"
        )

    def _load_model(self) -> None:
        """延遲載入 Whisper 模型

        Raises:
            WhisperModelError: whisper 套件未安裝或模型載入失敗
        """
        if self._model is not None:
            return

        try:
            import whisper
        except ImportError:
            raise WhisperModelError(
                "openai-whisper 套件未安裝。"
                "請執行: uv pip install openai-whisper"
            )

        try:
            logger.info(f"正在載入 Whisper-{self.model_size} 模型（首次需下載）...")
            self._model = whisper.load_model(
                self.model_size,
                device=self.device
            )
            logger.info(f"Whisper-{self.model_size} 模型載入完成")
        except Exception as e:
            raise WhisperModelError(f"Whisper 模型載入失敗: {e}") from e

    def transcribe(
        self,
        audio: np.ndarray,
        sr: int,
        language: str = "zh",
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """轉錄音頻為文字

        Args:
            audio: 降噪後的音頻陣列（float32, mono, 16kHz）
            sr: 取樣率（應為 16000）
            language: 指定語言代碼。"zh" 為中文，None 為自動偵測
            temperature: 解碼溫度。0.0 為 greedy decoding（最穩定）

        Returns:
            TranscriptionResult: 轉錄結果（含時間戳片段和信心度）

        Raises:
            WhisperModelError: 模型載入或推理失敗
        """
        self._load_model()

        try:
            # 確保音頻為 float32
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # 呼叫 Whisper 轉錄
            result = self._model.transcribe(
                audio,
                language=language,
                temperature=temperature,
                fp16=(self.device == "cuda"),  # GPU 使用半精度加速
                verbose=False,
            )

            # 解析片段與簡易角色分離 (A/B)
            segments = []
            total_logprob = 0.0
            total_tokens = 0
            
            current_speaker = "A"
            prev_end = 0.0
            full_text_parts = []

            for seg in result.get("segments", []):
                start = float(seg["start"])
                end = float(seg["end"])
                raw_text = seg["text"].strip()
                
                # 如果與上一段的間隔 > 0.8 秒，且不是第一段，則切換角色
                if segments and (start - prev_end) > 0.8:
                    current_speaker = "B" if current_speaker == "A" else "A"
                
                # 加上角色前綴
                speaker_text = f"[角色 {current_speaker}]: {raw_text}"
                full_text_parts.append(speaker_text)

                segments.append(TranscriptionSegment(
                    start=start,
                    end=end,
                    text=speaker_text,
                ))
                prev_end = end

                # 累積 log probability 用於計算整體信心度
                avg_logprob = seg.get("avg_logprob", -1.0)
                n_tokens = seg.get("tokens", [])
                total_logprob += avg_logprob * len(n_tokens)
                total_tokens += len(n_tokens)

            # 計算整體信心度（將 log probability 映射為 [0, 1]）
            if total_tokens > 0:
                avg_logprob = total_logprob / total_tokens
                # log probability 範圍約 [-inf, 0]，映射到 [0, 1]
                confidence = min(1.0, max(0.0, 1.0 + avg_logprob / 2.0))
            else:
                confidence = 0.0

            # 偵測語言
            detected_lang = result.get("language", language or "unknown")

            # 組合帶有角色的完整文字稿
            final_text = "\n".join(full_text_parts)

            transcript = TranscriptionResult(
                text=final_text,
                segments=segments,
                language=detected_lang,
                confidence=round(confidence, 4),
            )

            logger.info(
                f"轉錄完成: {transcript.word_count}字, "
                f"{len(segments)}片段, "
                f"語言={detected_lang}, "
                f"信心度={confidence:.2%}"
            )

            return transcript

        except Exception as e:
            raise WhisperModelError(f"Whisper 轉錄失敗: {e}") from e

    def is_model_loaded(self) -> bool:
        """檢查模型是否已載入"""
        return self._model is not None
