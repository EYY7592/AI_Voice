"""
語音降噪處理器
===============
使用 noisereduce 的 Spectral Gating 演算法進行自適應降噪。
"""
import numpy as np
import noisereduce as nr

from src.utils.logger import get_logger
from src.utils.exceptions import AudioDenoiseError

logger = get_logger("ai_voice.step1.denoiser")


class Denoiser:
    """語音降噪處理器

    基於 noisereduce 套件的 Spectral Gating 演算法，
    自動估算噪音 profile 並進行頻譜遮罩降噪。

    Attributes:
        prop_decrease: 降噪強度（0~1），1 為完全降噪
    """

    def __init__(self, prop_decrease: float = 0.8) -> None:
        """初始化

        Args:
            prop_decrease: 降噪強度，0 為不降噪，1 為完全降噪
                           建議範圍 0.6~0.9，預設 0.8
        """
        if not 0.0 <= prop_decrease <= 1.0:
            raise ValueError(f"prop_decrease 必須在 0~1 之間，收到: {prop_decrease}")
        self.prop_decrease = prop_decrease
        logger.info(f"Denoiser 初始化完成，降噪強度: {prop_decrease}")

    def denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """對音頻進行 Spectral Gating 自適應降噪

        自動從音頻中估算噪音 profile，然後應用頻譜遮罩降噪。

        Args:
            audio: 原始音頻陣列（float32）
            sr: 取樣率

        Returns:
            np.ndarray: 降噪後的音頻陣列

        Raises:
            AudioDenoiseError: 降噪處理失敗
        """
        if len(audio) == 0:
            logger.warning("收到空音頻，跳過降噪")
            return audio

        try:
            # 使用 noisereduce 的 Spectral Gating 降噪
            # stationary=True: 假設噪音是穩態的（適合電話線路噪音）
            denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                prop_decrease=self.prop_decrease,
                stationary=True,
                n_fft=2048,
                hop_length=512,
            )

            # 計算降噪前後的 RMS 變化（用於日誌）
            rms_before = float(np.sqrt(np.mean(audio ** 2)))
            rms_after = float(np.sqrt(np.mean(denoised ** 2)))
            reduction_db = 20 * np.log10(max(rms_after, 1e-10) / max(rms_before, 1e-10))

            logger.info(
                f"降噪完成: RMS 變化 {reduction_db:.1f}dB, "
                f"prop_decrease={self.prop_decrease}"
            )

            return denoised.astype(np.float32)

        except Exception as e:
            raise AudioDenoiseError(f"降噪處理失敗: {e}") from e

    def denoise_with_noise_sample(
        self,
        audio: np.ndarray,
        sr: int,
        noise_sample: np.ndarray
    ) -> np.ndarray:
        """使用指定的噪音樣本進行降噪

        當有已知的噪音片段（例如通話前的靜音段）時，
        可以更精確地進行降噪。

        Args:
            audio: 原始音頻陣列
            sr: 取樣率
            noise_sample: 噪音樣本陣列

        Returns:
            np.ndarray: 降噪後的音頻陣列
        """
        try:
            denoised = nr.reduce_noise(
                y=audio,
                sr=sr,
                y_noise=noise_sample,
                prop_decrease=self.prop_decrease,
                stationary=True,
            )
            logger.info("使用指定噪音樣本完成降噪")
            return denoised.astype(np.float32)

        except Exception as e:
            raise AudioDenoiseError(f"指定噪音樣本降噪失敗: {e}") from e

    def estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """估算音頻的信噪比（SNR）

        透過比較降噪前後的能量差異來估算 SNR。

        Args:
            audio: 原始音頻陣列
            sr: 取樣率

        Returns:
            float: 估計的 SNR（dB）
        """
        if len(audio) == 0:
            return 0.0

        # 降噪得到「信號」，差值近似為「噪音」
        denoised = self.denoise(audio, sr)
        noise = audio - denoised

        signal_power = float(np.mean(denoised ** 2))
        noise_power = float(np.mean(noise ** 2))

        if noise_power < 1e-10:
            return 60.0  # 非常乾淨的信號

        snr = 10 * np.log10(signal_power / noise_power)
        logger.info(f"估計 SNR: {snr:.1f}dB")
        return float(snr)
