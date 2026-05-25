"""
Agent 抽象基底類別
==================
定義所有 Agent 的統一介面和共用方法。
"""
from abc import ABC, abstractmethod
from typing import Any

from src.models import AgentResult
from src.utils.logger import get_logger

logger = get_logger("ai_voice.step3.base")


class BaseAgent(ABC):
    """Agent 抽象基底類別

    所有 Agent（聲紋/語義/記憶）都必須繼承此類，
    實作 load_model() 和 analyze() 方法。

    Attributes:
        name: Agent 名稱
        model_path: 模型權重檔案路徑
        is_loaded: 模型是否已載入
    """

    def __init__(self, name: str, model_path: str | None = None) -> None:
        """初始化

        Args:
            name: Agent 名稱
            model_path: 模型權重檔案路徑（None 時使用規則判斷）
        """
        self.name = name
        self.model_path = model_path
        self.is_loaded = False
        logger.info(f"Agent [{name}] 初始化，模型路徑: {model_path or '未指定（規則模式）'}")

    @abstractmethod
    def load_model(self) -> None:
        """載入模型權重

        子類別必須實作。若 model_path 為 None，
        則進入規則判斷模式（不需載入模型）。

        Raises:
            ModelLoadError: 權重檔案不存在或損壞
        """
        pass

    @abstractmethod
    def analyze(self, **kwargs) -> AgentResult:
        """執行分析

        子類別必須實作。接收特定類型的輸入，
        返回統一的 AgentResult。

        Returns:
            AgentResult

        Raises:
            AgentAnalysisError: 分析失敗
        """
        pass

    def _create_result(
        self,
        fraud_probability: float,
        confidence: float,
        signal_quality: float,
        details: dict,
        explanation: str,
    ) -> AgentResult:
        """建立標準化的 AgentResult

        Args:
            fraud_probability: 詐騙機率 [0, 1]
            confidence: 信心度 [0, 1]
            signal_quality: 信號品質 [0, 1]
            details: 詳細資訊
            explanation: 人類可讀說明

        Returns:
            AgentResult
        """
        return AgentResult(
            agent_name=self.name,
            fraud_probability=fraud_probability,
            confidence=confidence,
            signal_quality=signal_quality,
            details=details,
            explanation=explanation,
        )
