"""
特徵萃取器
==========
萃取三大類音頻特徵：通用聲學（MFCC/Mel）、韻律（Praat）、深度特徵（Wav2vec2）。
"""
import numpy as np
import librosa

from src.models import AudioFeatures, ProsodyFeatures
from src.utils.logger import get_logger
from src.utils.exceptions import FeatureExtractionError

# praat-parselmouth 韻律分析
try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False

logger = get_logger("ai_voice.step1.features")


class FeatureExtractor:
    """特徵萃取器（通用 + 韻律 + 深度）

    負責從降噪後的音頻中萃取所有必要特徵，
    供聲紋分析 Agent 及其他模組使用。

    Attributes:
        n_mfcc: MFCC 係數數量
        n_mels: Mel 頻帶數量
        use_wav2vec2: 是否提取 Wav2vec2 深度特徵
    """

    def __init__(
        self,
        n_mfcc: int = 40,
        n_mels: int = 128,
        use_wav2vec2: bool = False,
        wav2vec2_model_path: str = "facebook/wav2vec2-base"
    ) -> None:
        """初始化

        Args:
            n_mfcc: MFCC 係數數量，預設 40
            n_mels: Mel 頻帶數量，預設 128
            use_wav2vec2: 是否啟用 Wav2vec2 深度特徵提取
        """
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.use_wav2vec2 = use_wav2vec2
        self.wav2vec2_model_path = wav2vec2_model_path

        # Wav2vec2 模型延遲載入
        self._wav2vec2_model = None
        self._wav2vec2_processor = None

        if not HAS_PARSELMOUTH:
            logger.warning("parselmouth 未安裝，韻律特徵將使用替代方法")

        logger.info(
            f"FeatureExtractor 初始化完成: "
            f"MFCC={n_mfcc}, Mel={n_mels}, Wav2vec2={use_wav2vec2}"
        )

    def extract_all(self, audio: np.ndarray, sr: int) -> AudioFeatures:
        """萃取所有特徵（通用 + 韻律 + 深度）

        Args:
            audio: 降噪後的音頻陣列（float32, mono）
            sr: 取樣率

        Returns:
            AudioFeatures: 完整特徵集合

        Raises:
            FeatureExtractionError: 萃取過程失敗
        """
        if len(audio) == 0:
            raise FeatureExtractionError("無法對空音頻進行特徵萃取")

        try:
            # 通用聲學特徵
            mfcc = self._extract_mfcc(audio, sr)
            mel = self._extract_mel_spectrogram(audio, sr)
            zcr = librosa.feature.zero_crossing_rate(audio)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)

            # 韻律特徵
            prosody = self.extract_prosody(audio, sr)

            # 深度特徵（可選）
            wav2vec2_feat = None
            if self.use_wav2vec2:
                wav2vec2_feat = self.extract_wav2vec2(audio, sr)

            features = AudioFeatures(
                mfcc=mfcc,
                mel_spectrogram=mel,
                zcr=zcr,
                spectral_centroid=spectral_centroid,
                prosody=prosody,
                wav2vec2_features=wav2vec2_feat,
                duration=len(audio) / sr,
                snr_estimate=0.0,  # 由 Denoiser 單獨計算
            )

            logger.info(
                f"特徵萃取完成: MFCC{mfcc.shape}, "
                f"Mel{mel.shape}, 時長{features.duration:.2f}s"
            )
            return features

        except FeatureExtractionError:
            raise
        except Exception as e:
            raise FeatureExtractionError(f"特徵萃取失敗: {e}") from e

    def _extract_mfcc(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """萃取 MFCC 特徵

        Args:
            audio: 音頻陣列
            sr: 取樣率

        Returns:
            np.ndarray: (n_mfcc, T) MFCC 矩陣
        """
        mfcc = librosa.feature.mfcc(
            y=audio, sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512,
        )
        return mfcc

    def _extract_mel_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """萃取 Mel 頻譜圖

        Args:
            audio: 音頻陣列
            sr: 取樣率

        Returns:
            np.ndarray: (n_mels, T) Mel 頻譜矩陣（dB 尺度）
        """
        mel = librosa.feature.melspectrogram(
            y=audio, sr=sr,
            n_mels=self.n_mels,
            n_fft=2048,
            hop_length=512,
        )
        # 轉換為 dB 尺度
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db

    def extract_prosody(self, audio: np.ndarray, sr: int) -> ProsodyFeatures:
        """使用 parselmouth/Praat 萃取韻律特徵

        萃取 Jitter、Shimmer、HNR、F0 統計、語速、停頓模式、共振峰等
        AI 語音難以完美模擬的微觀特徵。

        Args:
            audio: 音頻陣列
            sr: 取樣率

        Returns:
            ProsodyFeatures: 韻律特徵集合
        """
        if not HAS_PARSELMOUTH:
            return self._extract_prosody_fallback(audio, sr)

        try:
            # 建立 Praat Sound 物件
            sound = parselmouth.Sound(audio, sampling_frequency=sr)

            # 萃取基頻（Pitch）
            pitch = call(sound, "To Pitch", 0.0, 75.0, 600.0)
            f0_values = pitch.selected_array["frequency"]
            f0_voiced = f0_values[f0_values > 0]  # 只取有聲段

            # F0 統計
            f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
            f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
            f0_range = float(np.ptp(f0_voiced)) if len(f0_voiced) > 0 else 0.0

            # 萃取 Point Process（用於 Jitter/Shimmer）
            point_process = call(
                sound, "To PointProcess (periodic, cc)",
                75.0, 600.0
            )

            # Jitter（基頻微抖動）
            jitter = call(
                point_process, "Get jitter (local)",
                0.0, 0.0, 0.0001, 0.02, 1.3
            )
            jitter = jitter if not np.isnan(jitter) else 0.0

            # Shimmer（振幅微抖動）
            shimmer = call(
                [sound, point_process], "Get shimmer (local)",
                0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
            )
            shimmer = shimmer if not np.isnan(shimmer) else 0.0

            # HNR（諧波噪聲比）
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0.0, 0.0)
            hnr = hnr if not np.isnan(hnr) else 0.0

            # 共振峰 (F1~F4)
            formants = self._extract_formants(sound)

            # 語速估算（基於有聲段比例和 F0 變化率）
            voiced_ratio = len(f0_voiced) / max(len(f0_values), 1)
            speaking_rate = voiced_ratio * sr / 160  # 近似音節率

            # 停頓分析（無聲段持續時間）
            pause_durations = self._extract_pauses(f0_values, pitch)

            prosody = ProsodyFeatures(
                jitter=float(jitter),
                shimmer=float(shimmer),
                hnr=float(hnr),
                f0_mean=f0_mean,
                f0_std=f0_std,
                f0_range=f0_range,
                speaking_rate=speaking_rate,
                pause_durations=pause_durations,
                formants=formants,
            )

            logger.info(
                f"韻律特徵: Jitter={jitter:.4f}, Shimmer={shimmer:.4f}, "
                f"HNR={hnr:.1f}dB, F0={f0_mean:.0f}±{f0_std:.0f}Hz"
            )
            return prosody

        except Exception as e:
            logger.warning(f"Praat 韻律分析失敗，使用替代方法: {e}")
            return self._extract_prosody_fallback(audio, sr)

    def _extract_formants(self, sound) -> list[float]:
        """萃取共振峰頻率 F1~F4

        Args:
            sound: parselmouth Sound 物件

        Returns:
            list[float]: [F1, F2, F3, F4] 頻率（Hz）
        """
        try:
            formant = call(sound, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            formants = []
            for i in range(1, 5):  # F1~F4
                f = call(formant, "Get mean", i, 0.0, 0.0, "hertz")
                formants.append(float(f) if not np.isnan(f) else 0.0)
            return formants
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

    def _extract_pauses(self, f0_values: np.ndarray, pitch) -> list[float]:
        """分析停頓模式（無聲段持續時間）

        Args:
            f0_values: F0 值陣列
            pitch: Praat Pitch 物件

        Returns:
            list[float]: 各停頓的持續時間（秒）
        """
        pauses = []
        time_step = pitch.time_step
        in_pause = False
        pause_start = 0

        for i, f0 in enumerate(f0_values):
            if f0 == 0 and not in_pause:
                # 停頓開始
                in_pause = True
                pause_start = i
            elif f0 > 0 and in_pause:
                # 停頓結束
                in_pause = False
                duration = (i - pause_start) * time_step
                if duration >= 0.05:  # 忽略極短停頓（< 50ms）
                    pauses.append(float(duration))

        return pauses

    def _extract_prosody_fallback(
        self, audio: np.ndarray, sr: int
    ) -> ProsodyFeatures:
        """韻律特徵替代方法（不依賴 parselmouth）

        使用 librosa 的基礎功能進行近似估算。

        Args:
            audio: 音頻陣列
            sr: 取樣率

        Returns:
            ProsodyFeatures: 近似韻律特徵
        """
        # 使用 librosa 的 pyin 估算 F0
        f0, voiced_flag, _ = librosa.pyin(
            audio, fmin=75, fmax=600,
            sr=sr, frame_length=2048
        )
        f0_voiced = f0[voiced_flag] if voiced_flag is not None else np.array([0.0])
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)] if len(f0_voiced) > 0 else np.array([0.0])

        f0_mean = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_std = float(np.std(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        f0_range = float(np.ptp(f0_voiced)) if len(f0_voiced) > 0 else 0.0

        # 語速近似（基於 RMS 能量的有聲段比例）
        rms = librosa.feature.rms(y=audio)[0]
        voiced_ratio = float(np.mean(rms > np.mean(rms) * 0.3))

        return ProsodyFeatures(
            jitter=0.0,   # 需要 parselmouth
            shimmer=0.0,  # 需要 parselmouth
            hnr=0.0,      # 需要 parselmouth
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_range=f0_range,
            speaking_rate=voiced_ratio * sr / 160,
            pause_durations=[],
            formants=[0.0, 0.0, 0.0, 0.0],
        )

    def extract_wav2vec2(self, audio: np.ndarray, sr: int):
        """使用 Wav2vec2-base 萃取深度特徵

        延遲載入模型，第一次呼叫時才下載/載入。

        Args:
            audio: 音頻陣列（16kHz）
            sr: 取樣率

        Returns:
            torch.Tensor: 隱藏層表示（最後一層）
        """
        try:
            import torch
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
        except ImportError:
            logger.warning("transformers/torch 未安裝，跳過 Wav2vec2 特徵")
            return None

        # 延遲載入模型
        if self._wav2vec2_model is None:
            logger.info(f"載入 Wav2vec2 模型: {self.wav2vec2_model_path}")
            self._wav2vec2_processor = Wav2Vec2Processor.from_pretrained(
                self.wav2vec2_model_path
            )
            self._wav2vec2_model = Wav2Vec2Model.from_pretrained(
                self.wav2vec2_model_path
            )
            self._wav2vec2_model.eval()
            logger.info("Wav2vec2 模型載入完成")

        # 前處理
        inputs = self._wav2vec2_processor(
            audio, sampling_rate=sr, return_tensors="pt", padding=True
        )

        # 推理
        with torch.no_grad():
            outputs = self._wav2vec2_model(**inputs)

        # 返回最後一層隱藏狀態序列
        hidden_states = outputs.last_hidden_state  # (1, T, 768)
        
        # 移除 batch 維度，返回 (T, 768)
        features = hidden_states.squeeze(0)
        
        logger.info(f"Wav2vec2 特徵提取完成: {features.shape}")
        return features
