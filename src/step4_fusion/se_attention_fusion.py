"""
SE-Attention 動態權重融合引擎
==============================
實作基於 Squeeze-and-Excitation (SE) 架構的動態特徵加權融合。
根據各 Agent 的信號品質、信心度和初步判定結果，動態分配偵測權重。
"""
import numpy as np
from typing import Any

from src.models import AgentResult, FusionResult
from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError

logger = get_logger("ai_voice.step4.fusion")


def get_torch_nn():
    """延遲導入 torch.nn"""
    import torch.nn as nn
    return nn


class SEAttentionMLP:
    """SE-Attention 權重預測網路 (與 Notebook 03 一致)"""
    def __new__(cls, *args, **kwargs):
        nn = get_torch_nn()
        
        class _SEAttentionMLP(nn.Module):
            def __init__(self, n_agents=3, features_per_agent=3):
                super().__init__()
                self.excitation = nn.Sequential(
                    nn.Linear(n_agents * features_per_agent, 32),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(16, n_agents),
                )

            def forward(self, x):
                return self.excitation(x)
        
        return _SEAttentionMLP(*args, **kwargs)


class SEAttentionFusion:
    """SE-Attention 動態融合引擎

    使用 Squeeze-and-Excitation 概念實作 Agent 層級的注意力機制。
    
    1. Squeeze：將各 Agent 的 [機率, 信心度, 信號品質] 壓縮為全球描述向量。
    2. Excitation：通過模型（或規則權重）產生各通道（Agent）的顯著性權重。
    3. Reweight：將權重套用到各 Agent 的結果。
    """

    def __init__(self, model_path: str | None = None) -> None:
        """初始化

        Args:
            model_path: 融合模型路徑（如 MLP 或權重矩陣）
        """
        self.model_path = model_path
        self._model = None
        self.is_loaded = False
        logger.info(f"SEAttentionFusion 初始化，模型路徑: {model_path or '未指定（使用自適應規則）'}")

    def load_model(self) -> None:
        """載入融合模型權重 (state_dict)"""
        if not self.model_path:
            return

        try:
            import torch
            # 1. 實例化模型結構
            self._model = SEAttentionMLP()
            
            # 2. 載入權重字典
            state_dict = torch.load(
                self.model_path, 
                map_location="cpu",
                weights_only=True
            )
            # 處理可能的嵌套
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
                
            self._model.load_state_dict(state_dict)
            self._model.eval()
            self.is_loaded = True
            logger.info(f"融合模型 (MLP) 載入成功: {self.model_path}")
        except Exception as e:
            logger.warning(f"融合模型載入失敗: {e}，將回退至自適應規則模式")
            self._model = None

    def fuse(self, agent_results: list[AgentResult]) -> FusionResult:
        """執行動態權重融合

        Args:
            agent_results: Agent A, B, C 的分析結果清單

        Returns:
            FusionResult
        """
        if not agent_results:
            return FusionResult(
                final_probability=0.0,
                dynamic_weights={},
                agent_results=[]
            )

        # 1. Squeeze: 提取各項特徵 (N_agents, 3)
        features = []
        for r in agent_results:
            features.append([
                r.fraud_probability,
                r.confidence,
                r.signal_quality
            ])
        features = np.array(features, dtype=np.float32)

        # 2. Excitation: 計算動態權重
        if self._model is not None:
            weights = self._model_inference(features)
        else:
            weights = self._rule_based_attention(agent_results)

        # 3. Reweight & Sum
        final_prob = 0.0
        weight_dict = {}
        for r, w in zip(agent_results, weights):
            final_prob += r.fraud_probability * w
            weight_dict[r.agent_name] = float(w)

        # 確保機率在 [0, 1]
        final_prob = float(np.clip(final_prob, 0.0, 1.0))

        logger.info(f"動態融合完成: 最終機率={final_prob:.4f}, 權重={weight_dict}")
        
        return FusionResult(
            final_probability=final_prob,
            dynamic_weights=weight_dict,
            agent_results=agent_results
        )

    def _rule_based_attention(self, agent_results: list[AgentResult]) -> np.ndarray:
        """基於規則的自適應注意力計算 (Excitation)

        核心邏輯：
        - 權重 = 基礎權重 * 信心度 * 信號品質
        - 若某 Agent 信心度為 0（如模型未載入），則權重歸零，由其他 Agent 瓜分。
        """
        n = len(agent_results)
        # 預設基礎權重（聲紋: 0.35, 語義: 0.40, 記憶: 0.25）
        base_weights = {
            "voiceprint": 0.35,
            "semantic": 0.40,
            "memory": 0.25
        }
        
        scores = []
        for r in agent_results:
            bw = base_weights.get(r.agent_name, 1.0 / n)
            # 有效權重考慮信心度和信號品質
            # 使用平方以非線性放大高品質信號的顯著性
            score = bw * (r.confidence * r.signal_quality + 1e-6)
            
            # 如果某個 Agent 的信心度極高且判定為詐騙，額外提升其權重
            if r.fraud_probability > 0.8 and r.confidence > 0.7:
                score *= 1.5
                
            scores.append(score)

        scores = np.array(scores)
        total = np.sum(scores)
        
        if total > 0:
            weights = scores / total
        else:
            weights = np.ones(n) / n # Fallback 為等權重
            
        return weights

    def _model_inference(self, features: np.ndarray) -> np.ndarray:
        """使用神經網路進行注意力推理"""
        import torch
        try:
            with torch.no_grad():
                # 展平特徵並增加 batch 維度
                x = torch.from_numpy(features.flatten()).float().unsqueeze(0)
                logits = self._model(x)
                # 使用 Softmax 確保權重和為 1
                weights = torch.softmax(logits, dim=-1)
                return weights.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"融合推理失敗: {e}，使用平均權重")
            return np.ones(len(features)) / len(features)
