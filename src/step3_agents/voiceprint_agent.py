"""
聲紋分析 Agent（Agent A）
==========================
雙層模型架構：
  Layer 1：LightGBM 韻律異常偵測（Jitter/Shimmer/HNR/F0/Formant）
  Layer 2：Wav2vec2+CNN 深偽偵測（判斷是否為 AI 合成語音）

兩層結果以加權方式融合為最終聲紋分析分數。
"""
import numpy as np

from src.step3_agents.base_agent import BaseAgent
from src.models import AudioFeatures, AgentResult, ProsodyFeatures
from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, AgentAnalysisError

logger = get_logger("ai_voice.step3.voiceprint")

# === 模型定義 (由 02_voiceprint_training.ipynb 移植) ===

class LGBMWrapper:
    """LightGBM 模型包裝器，提供與原本規則模式相容的 predict_proba 介面"""
    def __init__(self, booster):
        self.booster = booster
    
    def predict_proba(self, X):
        import numpy as np
        p1 = self.booster.predict(X)
        p0 = 1 - p1
        return np.column_stack([p0, p1])
    
    def predict(self, X):
        return (self.booster.predict(X) > 0.5).astype(int)

# 延遲導入 torch.nn 以避免全域依賴
class DeepfakeCNN:
    """Wav2vec2 特徵 + 3層 CNN 二元分類器"""
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn
        
        class _DeepfakeCNN(nn.Module):
            def __init__(self, input_dim=768, hidden_dim=256):
                super().__init__()
                self.conv_layers = nn.Sequential(
                    nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.3),
                    
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(0.3),
                    
                    nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, 1),
                )
            
            def forward(self, x):
                import torch
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                x = x.permute(0, 2, 1)
                x = self.conv_layers(x)
                x = x.squeeze(-1)
                return self.classifier(x)
        
        return _DeepfakeCNN(*args, **kwargs)


class VoiceprintAgent(BaseAgent):
    """Agent A：聲紋分析（韻律 + 深偽雙層模型）

    Layer 1：使用 LightGBM 對韻律特徵進行異常偵測
    Layer 2：使用 Wav2vec2 特徵 + CNN 判斷 AI 合成語音

    當模型未載入時，自動使用基於規則的暫態判斷。
    """

    # 聲紋分析的規則閾值（基於文獻值）
    THRESHOLDS = {
        "jitter_low": 0.003,   # AI 語音 Jitter 通常 < 0.3%
        "jitter_high": 0.04,   # 正常語音 Jitter 上限
        "shimmer_low": 0.01,   # AI 語音 Shimmer 通常 < 1%
        "hnr_high": 30.0,      # AI 語音 HNR 異常高
        "f0_cv_low": 0.08,     # AI 語音 F0 變異係數過低
        "f0_cv_high": 0.35,    # 正常語音 F0 變異係數上限
        "pause_std_low": 0.03, # AI 語音停頓過於規律
    }

    def __init__(
        self,
        prosody_model_path: str | None = None,
        deepfake_model_path: str | None = None
    ) -> None:
        """初始化

        Args:
            prosody_model_path: LightGBM 韻律模型路徑（.pkl）
            deepfake_model_path: Wav2vec2+CNN 深偽模型路徑（.pt）
        """
        super().__init__(name="voiceprint")
        self.prosody_model_path = prosody_model_path
        self.deepfake_model_path = deepfake_model_path
        self._prosody_model = None   # LightGBM 模型
        self._deepfake_model = None  # CNN 模型

    def load_model(self) -> None:
        """載入聲紋分析模型

        嘗試載入 LightGBM 韻律模型和 CNN 深偽模型。
        若模型不存在，會記錄警告但不拋出例外（使用規則模式）。
        """
        # 載入韻律 LightGBM 模型
        if self.prosody_model_path:
            try:
                import pickle
                import io

                # 自訂 Unpickler：將 pickle 中的 __main__.LGBMWrapper
                # 重新導向至當前模組中定義的 LGBMWrapper 類別
                class _Unpickler(pickle.Unpickler):
                    def find_class(self_, module, name):
                        if name == "LGBMWrapper":
                            return LGBMWrapper
                        return super().find_class(module, name)

                with open(self.prosody_model_path, "rb") as f:
                    self._prosody_model = _Unpickler(f).load()
                logger.info(f"LightGBM 韻律模型載入成功: {self.prosody_model_path}")
            except FileNotFoundError:
                logger.warning(f"韻律模型不存在: {self.prosody_model_path}，使用規則模式")
            except Exception as e:
                logger.warning(f"韻律模型載入失敗: {e}，使用規則模式")

        # 載入深偽 CNN 模型
        if self.deepfake_model_path:
            try:
                import torch
                # 實例化模型結構
                # 這裡調用 DeepfakeCNN() 實際上會回傳 _DeepfakeCNN 實體
                self._deepfake_model = DeepfakeCNN()
                # 載入權重 (state_dict)
                state_dict = torch.load(
                    self.deepfake_model_path,
                    map_location="cpu",
                    weights_only=True,
                )
                # 處理可能的 state_dict 嵌套
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                
                self._deepfake_model.load_state_dict(state_dict)
                self._deepfake_model.eval()
                logger.info(f"深偽 CNN 模型載入成功: {self.deepfake_model_path}")
            except FileNotFoundError:
                logger.warning(f"深偽模型不存在: {self.deepfake_model_path}，使用規則模式")
                self._deepfake_model = None
            except Exception as e:
                logger.warning(f"深偽模型載入失敗: {e}，使用規則模式")
                self._deepfake_model = None

        self.is_loaded = (
            self._prosody_model is not None or
            self._deepfake_model is not None
        )

    def analyze(self, audio_features: AudioFeatures) -> AgentResult:
        """綜合聲紋分析

        先分別執行韻律分析和深偽偵測，再以加權方式融合。

        Args:
            audio_features: Step1 萃取的完整音頻特徵

        Returns:
            AgentResult: 聲紋分析結果
        """
        try:
            # Layer 1：韻律分析
            prosody_score, prosody_details = self._analyze_prosody(
                audio_features.prosody
            )

            # Layer 2：深偽偵測
            deepfake_score, deepfake_details = self._analyze_deepfake(
                audio_features
            )

            # 融合兩層分數
            # 韻律 40% + 深偽 60%（深偽偵測更可靠）
            if deepfake_score is not None and prosody_score is not None:
                combined_score = 0.4 * prosody_score + 0.6 * deepfake_score
                confidence = 0.85
            elif prosody_score is not None:
                combined_score = prosody_score
                confidence = 0.6  # 只有韻律，信心度較低
            elif deepfake_score is not None:
                combined_score = deepfake_score
                confidence = 0.80
            else:
                combined_score = 0.0
                confidence = 0.0

            # 信號品質
            signal_quality = self._compute_signal_quality(audio_features)

            # 合併詳細資訊
            details = {**prosody_details, **deepfake_details}
            details["prosody_score"] = prosody_score
            details["deepfake_score"] = deepfake_score

            # 生成解釋
            explanation = self._generate_explanation(
                prosody_score, deepfake_score, prosody_details
            )

            return self._create_result(
                fraud_probability=combined_score,
                confidence=confidence,
                signal_quality=signal_quality,
                details=details,
                explanation=explanation,
            )

        except Exception as e:
            raise AgentAnalysisError(f"聲紋分析失敗: {e}") from e

    def _analyze_prosody(
        self, prosody: ProsodyFeatures
    ) -> tuple[float | None, dict]:
        """韻律分析（Layer 1）

        使用 LightGBM 模型或規則判斷。

        Returns:
            (異常分數 [0,1], 詳細資訊)
        """
        details = {
            "jitter": prosody.jitter,
            "shimmer": prosody.shimmer,
            "hnr": prosody.hnr,
            "f0_mean": prosody.f0_mean,
            "f0_std": prosody.f0_std,
            "f0_range": prosody.f0_range,
        }

        # 如果有 LightGBM 模型，使用模型推理
        if self._prosody_model is not None:
            # Step1 的 ProsodyFeatures.to_feature_vector() 回傳 20 維向量
            feature_vec = prosody.to_feature_vector().reshape(1, -1)
            # 使用包裝器的 predict_proba
            score = float(self._prosody_model.predict_proba(feature_vec)[0][1])
            details["mode"] = "model"
            details["model_score"] = score
            return score, details

        # 規則模式
        return self._prosody_rule_based(prosody, details)

    def _prosody_rule_based(
        self, prosody: ProsodyFeatures, details: dict
    ) -> tuple[float, dict]:
        """韻律規則判斷（模型未載入時使用）

        基於聲學文獻的 AI 語音特徵：
        - Jitter/Shimmer 過低 → 合成語音微擾動不足
        - HNR 過高 → 合成語音過於乾淨
        - F0 變異過低 → 語調過於平穩
        - 停頓過於規律 → 生成節奏缺乏自然變化

        Returns:
            (異常分數 [0,1], 詳細資訊)
        """
        score = 0.0
        flags = []

        th = self.THRESHOLDS

        # Jitter 偵測
        if 0 < prosody.jitter < th["jitter_low"]:
            score += 0.25
            flags.append(f"Jitter過低({prosody.jitter:.4f})")
        elif prosody.jitter > th["jitter_high"]:
            score += 0.1  # 異常高也可疑
            flags.append(f"Jitter異常高({prosody.jitter:.4f})")

        # Shimmer 偵測
        if 0 < prosody.shimmer < th["shimmer_low"]:
            score += 0.2
            flags.append(f"Shimmer過低({prosody.shimmer:.4f})")

        # HNR 偵測
        if prosody.hnr > th["hnr_high"]:
            score += 0.2
            flags.append(f"HNR過高({prosody.hnr:.1f}dB)")

        # F0 變異係數偵測
        if prosody.f0_mean > 0:
            f0_cv = prosody.f0_std / prosody.f0_mean
            if f0_cv < th["f0_cv_low"]:
                score += 0.2
                flags.append(f"F0變異過低(CV={f0_cv:.3f})")
            details["f0_cv"] = f0_cv

        # 停頓規律性偵測
        if prosody.pause_durations and len(prosody.pause_durations) > 2:
            pause_std = float(np.std(prosody.pause_durations))
            if pause_std < th["pause_std_low"]:
                score += 0.15
                flags.append(f"停頓過規律(std={pause_std:.4f})")
            details["pause_std"] = pause_std

        details["mode"] = "rule"
        details["flags"] = flags

        return min(1.0, score), details

    def _analyze_deepfake(
        self, features: AudioFeatures
    ) -> tuple[float | None, dict]:
        """深偽偵測（Layer 2）

        使用 Wav2vec2 特徵 + CNN 模型或返回 None。

        Returns:
            (合成語音分數 [0,1] 或 None, 詳細資訊)
        """
        details = {}

        if self._deepfake_model is not None and features.wav2vec2_features is not None:
            try:
                import torch
                with torch.no_grad():
                    feat = features.wav2vec2_features
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(0)
                    output = self._deepfake_model(feat)
                    score = float(torch.sigmoid(output).item())
                details["deepfake_mode"] = "model"
                return score, details
            except Exception as e:
                logger.warning(f"深偽模型推理失敗: {e}")

        details["deepfake_mode"] = "unavailable"
        return None, details

    def _compute_signal_quality(self, features: AudioFeatures) -> float:
        """計算信號品質

        基於 SNR、音頻時長和頻譜完整度。

        Returns:
            品質分數 [0, 1]
        """
        scores = []

        # SNR 品質（0~40dB 線性映射）
        snr_score = min(1.0, max(0.0, features.snr_estimate / 40.0))
        scores.append(snr_score)

        # 時長品質（太短或太長扣分）
        if features.duration < 1.0:
            dur_score = features.duration  # 線性衰減
        elif features.duration > 300:
            dur_score = 0.5  # 超長音頻品質下降
        else:
            dur_score = 1.0
        scores.append(dur_score)

        # MFCC 覆蓋率（有效幀比例）
        mfcc_coverage = float(np.mean(np.abs(features.mfcc) > 1e-6))
        scores.append(min(1.0, mfcc_coverage))

        return float(np.mean(scores))

    def _generate_explanation(
        self,
        prosody_score: float | None,
        deepfake_score: float | None,
        details: dict,
    ) -> str:
        """生成人類可讀的分析說明"""
        parts = []

        if prosody_score is not None:
            if prosody_score > 0.6:
                parts.append(f"韻律特徵高度異常（分數 {prosody_score:.2f}）")
            elif prosody_score > 0.3:
                parts.append(f"韻律特徵有部分異常（分數 {prosody_score:.2f}）")
            else:
                parts.append(f"韻律特徵在正常範圍（分數 {prosody_score:.2f}）")

        flags = details.get("flags", [])
        if flags:
            parts.append("異常項：" + "、".join(flags[:3]))

        if deepfake_score is not None:
            if deepfake_score > 0.7:
                parts.append(f"深偽偵測：高度可疑合成語音（{deepfake_score:.2f}）")
            else:
                parts.append(f"深偽偵測：可能為自然語音（{deepfake_score:.2f}）")
        else:
            parts.append("深偽偵測：模型未載入")

        return "；".join(parts) if parts else "無法生成分析說明"
