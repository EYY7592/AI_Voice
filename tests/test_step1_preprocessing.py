"""
Step1 特徵萃取 + 降噪模組單元測試
===================================
測試 AudioLoader, Denoiser, FeatureExtractor 三個核心類別。
"""
import os
import tempfile

import numpy as np
import pytest
import soundfile as sf

from src.step1_preprocessing.audio_loader import AudioLoader, SUPPORTED_FORMATS
from src.step1_preprocessing.denoiser import Denoiser
from src.step1_preprocessing.feature_extractor import FeatureExtractor
from src.utils.exceptions import AudioLoadError, AudioDenoiseError, FeatureExtractionError


# ============================================================
# AudioLoader 測試
# ============================================================

class TestAudioLoader:
    """AudioLoader 載入器測試"""

    @pytest.fixture
    def loader(self):
        """建立 AudioLoader 實例"""
        return AudioLoader(target_sr=16000)

    @pytest.fixture
    def wav_file(self, sample_audio):
        """建立暫時 WAV 測試檔案"""
        audio, sr = sample_audio
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, sr)
            yield f.name
        os.unlink(f.name)

    def test_load_wav(self, loader, wav_file):
        """測試 WAV 檔案載入"""
        audio, sr = loader.load(wav_file)
        assert sr == 16000
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_load_nonexistent(self, loader):
        """測試不存在的檔案 → 拋出 AudioLoadError"""
        with pytest.raises(AudioLoadError, match="不存在"):
            loader.load("not_exist.wav")

    def test_load_unsupported_format(self, loader):
        """測試不支援的格式 → 拋出 AudioLoadError"""
        tmp_path = os.path.join(tempfile.gettempdir(), "test_unsupported.txt")
        try:
            with open(tmp_path, "wb") as f:
                f.write(b"not audio")
            with pytest.raises(AudioLoadError, match="不支援"):
                loader.load(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_segment_short_audio(self, loader, sample_audio):
        """測試短音頻不需分段"""
        audio, sr = sample_audio  # 1 秒
        segments = loader.segment(audio, sr, max_duration=30.0)
        assert len(segments) == 1

    def test_segment_long_audio(self, loader, long_audio):
        """測試長音頻正確分段"""
        audio, sr = long_audio  # 60 秒
        segments = loader.segment(audio, sr, max_duration=30.0, overlap=0.0)
        assert len(segments) == 2

    def test_segment_with_overlap(self, loader, long_audio):
        """測試重疊分段"""
        audio, sr = long_audio  # 60 秒
        segments = loader.segment(audio, sr, max_duration=30.0, overlap=0.5)
        # 30 秒片段，50% 重疊 → 步長 15 秒，0/15/30/45 → 4 段
        assert len(segments) >= 3

    def test_remove_silence(self, loader, sample_audio):
        """測試靜音去除"""
        audio, sr = sample_audio
        # 加入 0.5 秒靜音
        silence = np.zeros(int(0.5 * sr), dtype=np.float32)
        audio_with_silence = np.concatenate([silence, audio, silence])
        result = loader.remove_silence(audio_with_silence, sr)
        # 去除靜音後應該變短
        assert len(result) < len(audio_with_silence)

    def test_remove_silence_all_silent(self, loader, empty_audio):
        """測試全靜音音頻不會崩潰"""
        audio, sr = empty_audio
        result = loader.remove_silence(audio, sr)
        assert len(result) > 0  # 應返回原始音頻

    def test_get_duration(self, loader, sample_audio):
        """測試時長計算"""
        audio, sr = sample_audio
        duration = loader.get_duration(audio, sr)
        assert duration == pytest.approx(1.0, abs=0.01)

    def test_supported_formats(self):
        """測試支援格式清單"""
        assert ".wav" in SUPPORTED_FORMATS
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".flac" in SUPPORTED_FORMATS


# ============================================================
# Denoiser 測試
# ============================================================

class TestDenoiser:
    """Denoiser 降噪器測試"""

    @pytest.fixture
    def denoiser(self):
        """建立 Denoiser 實例"""
        return Denoiser(prop_decrease=0.8)

    def test_init_valid(self):
        """測試有效初始化"""
        d = Denoiser(prop_decrease=0.5)
        assert d.prop_decrease == 0.5

    def test_init_invalid(self):
        """測試無效降噪強度 → 拋出 ValueError"""
        with pytest.raises(ValueError):
            Denoiser(prop_decrease=1.5)
        with pytest.raises(ValueError):
            Denoiser(prop_decrease=-0.1)

    def test_denoise_normal(self, denoiser, sample_audio):
        """測試正常降噪"""
        audio, sr = sample_audio
        # 加入噪音
        noisy = audio + 0.3 * np.random.randn(len(audio)).astype(np.float32)
        denoised = denoiser.denoise(noisy, sr)
        assert len(denoised) == len(noisy)
        assert denoised.dtype == np.float32

    def test_denoise_empty(self, denoiser):
        """測試空音頻降噪不崩潰"""
        audio = np.array([], dtype=np.float32)
        result = denoiser.denoise(audio, 16000)
        assert len(result) == 0

    def test_denoise_preserves_signal(self, denoiser, sample_audio):
        """測試降噪不會過度損失信號"""
        audio, sr = sample_audio
        denoised = denoiser.denoise(audio, sr)
        # 降噪前後的相關性應該很高（噪音少的音頻）
        correlation = np.corrcoef(audio[:len(denoised)], denoised[:len(audio)])[0, 1]
        assert correlation > 0.5  # 至少有中度相關

    def test_estimate_snr(self, denoiser, sample_audio):
        """測試 SNR 估算返回有效數值"""
        audio, sr = sample_audio
        snr = denoiser.estimate_snr(audio, sr)
        assert isinstance(snr, float)
        assert not np.isnan(snr)  # 確保不是 NaN


# ============================================================
# FeatureExtractor 測試
# ============================================================

class TestFeatureExtractor:
    """FeatureExtractor 特徵萃取器測試"""

    @pytest.fixture
    def extractor(self):
        """建立 FeatureExtractor 實例（不啟用 Wav2vec2）"""
        return FeatureExtractor(n_mfcc=40, n_mels=128, use_wav2vec2=False)

    def test_extract_all_shape(self, extractor, sample_audio):
        """測試完整特徵萃取的形狀"""
        audio, sr = sample_audio
        features = extractor.extract_all(audio, sr)
        assert features.mfcc.shape[0] == 40    # n_mfcc
        assert features.mel_spectrogram.shape[0] == 128  # n_mels
        assert features.zcr.shape[0] == 1
        assert features.spectral_centroid.shape[0] == 1
        assert features.duration == pytest.approx(1.0, abs=0.01)

    def test_extract_all_empty_raises(self, extractor):
        """測試空音頻 → 拋出 FeatureExtractionError"""
        with pytest.raises(FeatureExtractionError):
            extractor.extract_all(np.array([], dtype=np.float32), 16000)

    def test_extract_prosody(self, extractor, sample_audio):
        """測試韻律特徵萃取"""
        audio, sr = sample_audio
        prosody = extractor.extract_prosody(audio, sr)
        # 型別檢查
        assert isinstance(prosody.jitter, float)
        assert isinstance(prosody.shimmer, float)
        assert isinstance(prosody.hnr, float)
        assert isinstance(prosody.f0_mean, float)
        assert len(prosody.formants) == 4

    def test_extract_prosody_short_audio(self, extractor, short_audio):
        """測試極短音頻的韻律萃取不崩潰"""
        audio, sr = short_audio  # 0.05 秒
        prosody = extractor.extract_prosody(audio, sr)
        assert isinstance(prosody, type(prosody))

    def test_prosody_feature_vector(self, extractor, sample_audio):
        """測試韻律特徵向量形狀"""
        audio, sr = sample_audio
        prosody = extractor.extract_prosody(audio, sr)
        vec = prosody.to_feature_vector()
        assert vec.shape == (20,)
        assert not np.any(np.isnan(vec))

    def test_wav2vec2_not_loaded(self, extractor):
        """測試未啟用 Wav2vec2 時不載入模型"""
        assert extractor._wav2vec2_model is None
        assert extractor.use_wav2vec2 is False

    def test_extract_all_no_wav2vec2(self, extractor, sample_audio):
        """測試不啟用 Wav2vec2 時 wav2vec2_features 為 None"""
        audio, sr = sample_audio
        features = extractor.extract_all(audio, sr)
        assert features.wav2vec2_features is None
