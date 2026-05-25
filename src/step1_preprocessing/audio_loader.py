"""
音頻載入與前處理器
=================
支援多種音頻格式的載入、重取樣、靜音去除與分段處理。
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path

from src.utils.logger import get_logger
from src.utils.exceptions import AudioLoadError

# 支援的音頻格式清單
SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma"}

logger = get_logger("ai_voice.step1.loader")


class AudioLoader:
    """音頻載入與前處理器

    負責將各種格式的音頻檔案載入為統一的 numpy 陣列，
    並進行重取樣、靜音去除等前處理。

    Attributes:
        target_sr: 目標取樣率（預設 16kHz，Whisper 與多數模型的標準）
    """

    def __init__(self, target_sr: int = 16000) -> None:
        """初始化

        Args:
            target_sr: 目標取樣率（Hz），預設 16000
        """
        self.target_sr = target_sr
        logger.info(f"AudioLoader 初始化完成，目標取樣率: {target_sr}Hz")

    def load(self, file_path: str) -> tuple[np.ndarray, int]:
        """載入音頻檔案並重取樣為目標取樣率的單聲道

        Args:
            file_path: 音頻檔案路徑（支援 WAV/MP3/FLAC/OGG/M4A/WMA）

        Returns:
            tuple: (audio_array, sample_rate)
                - audio_array: float32 單聲道音頻陣列
                - sample_rate: 取樣率（等於 self.target_sr）

        Raises:
            AudioLoadError: 檔案不存在、格式不支援或讀取失敗
        """
        path = Path(file_path)

        # 檢查檔案存在性
        if not path.exists():
            raise AudioLoadError(f"音頻檔案不存在: {file_path}")

        # 檢查格式支援
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_FORMATS:
            raise AudioLoadError(
                f"不支援的音頻格式: {suffix}。"
                f"支援格式: {', '.join(sorted(SUPPORTED_FORMATS))}"
            )

        try:
            # 使用 librosa 載入並自動重取樣為目標取樣率
            audio, sr = librosa.load(
                str(path),
                sr=self.target_sr,
                mono=True,  # 強制轉為單聲道
                dtype=np.float32
            )
            logger.info(
                f"成功載入音頻: {path.name}, "
                f"時長: {len(audio) / sr:.2f}s, "
                f"取樣率: {sr}Hz"
            )
            return audio, sr

        except Exception as e:
            raise AudioLoadError(f"載入音頻失敗 ({path.name}): {e}") from e

    def segment(
        self,
        audio: np.ndarray,
        sr: int,
        max_duration: float = 30.0,
        overlap: float = 0.5
    ) -> list[np.ndarray]:
        """將超長音頻切割為可處理的固定長度片段

        Args:
            audio: 音頻陣列
            sr: 取樣率
            max_duration: 每個片段的最大秒數
            overlap: 片段間的重疊比例（0~1），用於避免邊界資訊丟失

        Returns:
            list[np.ndarray]: 音頻片段清單
        """
        total_samples = len(audio)
        segment_samples = int(max_duration * sr)

        # 如果音頻不需要分段，直接返回
        if total_samples <= segment_samples:
            return [audio]

        # 計算步長（考慮重疊）
        step_samples = int(segment_samples * (1 - overlap))
        segments = []

        for start in range(0, total_samples, step_samples):
            end = min(start + segment_samples, total_samples)
            segment = audio[start:end]

            # 過濾掉太短的尾段（小於 0.5 秒）
            if len(segment) >= int(0.5 * sr):
                segments.append(segment)

        logger.info(
            f"音頻分段完成: {len(segments)} 段, "
            f"每段最長 {max_duration}s, 重疊 {overlap * 100:.0f}%"
        )
        return segments

    def remove_silence(
        self,
        audio: np.ndarray,
        sr: int,
        top_db: int = 30
    ) -> np.ndarray:
        """偵測並去除靜音片段

        使用 librosa 的非靜音區間偵測，移除頭尾及中間的靜音。

        Args:
            audio: 音頻陣列
            sr: 取樣率
            top_db: 靜音偵測閾值（dB），數值越小越敏感

        Returns:
            np.ndarray: 去除靜音後的音頻陣列
        """
        # 偵測非靜音區間
        intervals = librosa.effects.split(audio, top_db=top_db)

        if len(intervals) == 0:
            logger.warning("音頻全為靜音，返回原始音頻")
            return audio

        # 拼接所有非靜音區間
        non_silent = np.concatenate([audio[start:end] for start, end in intervals])

        removed_duration = (len(audio) - len(non_silent)) / sr
        logger.info(f"靜音去除完成: 移除 {removed_duration:.2f}s 靜音")

        return non_silent

    def get_duration(self, audio: np.ndarray, sr: int) -> float:
        """計算音頻時長

        Args:
            audio: 音頻陣列
            sr: 取樣率

        Returns:
            float: 時長（秒）
        """
        return len(audio) / sr
