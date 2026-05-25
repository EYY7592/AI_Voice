"""
語義分析 Agent（Agent B）
==========================
使用 BERT-base-chinese 進行詐騙語義分類。
當模型未載入時，使用中文詐騙關鍵詞規則判斷。
"""
import re
from typing import Any

import numpy as np

from src.step3_agents.base_agent import BaseAgent
from src.models import TranscriptionResult, AgentResult
from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, AgentAnalysisError

logger = get_logger("ai_voice.step3.semantic")


class SemanticAgent(BaseAgent):
    """Agent B：語義詐騙分析

    使用 BERT-base-chinese fine-tuned 模型進行文字稿詐騙分類。
    或使用基於關鍵詞+語義規則的暫態判斷。
    """

    # 詐騙類型及其對應的關鍵詞庫
    FRAUD_PATTERNS = {
        "冒充公檢法": {
            "keywords": [
                "公安局", "檢察院", "法院", "警官", "警察", "公安",
                "涉嫌", "洗錢", "犯罪", "逮捕", "拘留", "強制措施",
                "安全帳戶", "資金清查", "案件", "傳票", "通緝",
            ],
            "patterns": [
                r"你的?(銀行|帳[戶號]|社保|醫保).{0,10}(涉嫌|異常|被盜|凍結)",
                r"(案件|案號|通緝).{0,10}(有關|涉及|牽涉)",
                r"(資金|錢款|存款).{0,10}轉(入|到|至).{0,10}(安全|指定)",
            ],
            "severity": 0.95,
        },
        "冒充客服退款": {
            "keywords": [
                "退款", "賠償", "理賠", "退貨", "訂單", "快遞",
                "轉帳", "匯款", "驗證碼", "銀行卡",
                "客服", "官方", "平台",
            ],
            "patterns": [
                r"(訂單|商品|快遞).{0,10}(問題|異常|損壞|丟失)",
                r"(退款|退貨|賠償).{0,10}(帳[戶號]|銀行卡|驗證碼)",
            ],
            "severity": 0.85,
        },
        "投資詐騙": {
            "keywords": [
                "投資", "理財", "股票", "基金", "收益", "利潤",
                "穩賺", "高報酬", "日收益", "月收益",
                "保本", "無風險", "內部消息", "老師",
            ],
            "patterns": [
                r"(日|月|年).{0,5}(收益|回報|報酬).{0,10}[\d]+%",
                r"(穩賺|保本|無風險|零風險).{0,10}(投資|理財)",
            ],
            "severity": 0.80,
        },
        "假冒親友": {
            "keywords": [
                "急用", "借我", "出事了", "住院", "手術",
                "學費", "罰款", "保釋金",
            ],
            "patterns": [
                r"(我是|這是).{0,5}(你[的兒女朋友]|爸|媽|哥|姐|弟|妹)",
                r"(急|趕緊|馬上|立刻).{0,10}(轉|匯|打).{0,10}(錢|款)",
            ],
            "severity": 0.75,
        },
        "中獎通知": {
            "keywords": [
                "中獎", "大獎", "獎金", "活動", "抽獎",
                "手續費", "稅款", "保證金",
            ],
            "patterns": [
                r"恭喜.{0,10}(中獎|獲得|贏得)",
                r"(領取|兌換).{0,10}(先|需要).{0,10}(繳|支付|匯)",
            ],
            "severity": 0.70,
        },
    }

    # 通用詐騙話術特徵
    PRESSURE_KEYWORDS = [
        "立刻", "馬上", "立即", "趕緊", "千萬不要", "不要告訴",
        "保密", "否則", "後果自負", "最後機會", "限時",
    ]

    def __init__(self, model_path: str | None = None) -> None:
        """初始化

        Args:
            model_path: BERT fine-tuned 模型路徑（.pt / 目錄）
        """
        super().__init__(name="semantic", model_path=model_path)
        self._model = None
        self._tokenizer = None

    def load_model(self) -> None:
        """載入 BERT 語義模型"""
        if not self.model_path:
            logger.info("未指定語義模型路徑，使用關鍵詞規則模式")
            return

        try:
            from transformers import AutoTokenizer, BertForSequenceClassification
            import torch

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self._model = BertForSequenceClassification.from_pretrained(self.model_path)
            self._model.eval()
            self.is_loaded = True
            logger.info(f"BERT 語義模型載入成功: {self.model_path}")

        except FileNotFoundError:
            logger.warning(f"語義模型不存在: {self.model_path}，使用規則模式")
        except Exception as e:
            logger.warning(f"語義模型載入失敗: {e}，使用規則模式")

    def analyze(self, transcript: TranscriptionResult) -> AgentResult:
        """分析轉錄文字的詐騙語義

        Args:
            transcript: Step2 的轉錄結果

        Returns:
            AgentResult（詐騙分類 + 關鍵詞 + 類型）
        """
        try:
            text = transcript.text

            if not text or not text.strip():
                return self._create_result(
                    fraud_probability=0.0,
                    confidence=0.0,
                    signal_quality=0.0,
                    details={"error": "文字稿為空"},
                    explanation="無文字可供分析",
                )

            # BERT 模型推理
            if self._model is not None and self._tokenizer is not None:
                return self._model_inference(text, transcript)

            # 規則模式
            return self._rule_based_analysis(text, transcript)

        except Exception as e:
            raise AgentAnalysisError(f"語義分析失敗: {e}") from e

    def _model_inference(
        self, text: str, transcript: TranscriptionResult
    ) -> AgentResult:
        """BERT 模型推理"""
        import torch

        # 截斷至 512 tokens
        inputs = self._tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512, padding=True
        )

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            # 模型標籤映射：label 0 = 詐騙，label 1 = 正常
            fraud_prob = float(probs[0][0])
            confidence = float(torch.max(probs).item())

        # 補充關鍵詞分析（取得詐騙子類型）
        fraud_type, _ = self._classify_fraud_type_scored(text)
        matched_keywords = self.extract_fraud_keywords(text)

        signal_quality = self._compute_signal_quality(transcript)

        return self._create_result(
            fraud_probability=fraud_prob,
            confidence=confidence,
            signal_quality=signal_quality,
            details={
                "mode": "model",
                "fraud_type": fraud_type,
                "keywords": matched_keywords,
            },
            explanation=f"BERT 模型判定詐騙機率 {fraud_prob:.1%}，類型: {fraud_type}",
        )

    def _rule_based_analysis(
        self, text: str, transcript: TranscriptionResult
    ) -> AgentResult:
        """基於關鍵詞 + 正則的規則分析"""
        # 分類詐騙類型
        fraud_type, max_score = self._classify_fraud_type_scored(text)
        matched_keywords = self.extract_fraud_keywords(text)

        # 壓力話術加分
        pressure_count = sum(1 for kw in self.PRESSURE_KEYWORDS if kw in text)
        pressure_bonus = min(0.15, pressure_count * 0.03)

        # 最終分數
        final_score = min(1.0, max_score + pressure_bonus)

        # 信心度（基於命中數量）
        total_hits = len(matched_keywords) + pressure_count
        confidence = min(0.9, max(0.3, total_hits * 0.06))

        signal_quality = self._compute_signal_quality(transcript)

        return self._create_result(
            fraud_probability=final_score,
            confidence=confidence,
            signal_quality=signal_quality,
            details={
                "mode": "rule",
                "fraud_type": fraud_type,
                "keywords": matched_keywords,
                "pressure_keywords_count": pressure_count,
                "keyword_hits": len(matched_keywords),
            },
            explanation=self._build_explanation(
                fraud_type, final_score, matched_keywords, pressure_count
            ),
        )

    def _classify_fraud_type_scored(self, text: str) -> tuple[str, float]:
        """帶分數的詐騙類型分類

        Returns:
            (詐騙類型, 匹配分數)
        """
        best_type = "未知"
        best_score = 0.0

        for fraud_type, config in self.FRAUD_PATTERNS.items():
            score = 0.0

            # 關鍵詞命中
            keyword_hits = sum(1 for kw in config["keywords"] if kw in text)
            keyword_score = min(0.5, keyword_hits * 0.05)

            # 正則模式命中
            pattern_hits = sum(
                1 for p in config["patterns"]
                if re.search(p, text)
            )
            pattern_score = min(0.5, pattern_hits * 0.15)

            score = (keyword_score + pattern_score) * config["severity"]

            if score > best_score:
                best_score = score
                best_type = fraud_type

        return best_type, best_score

    def classify_fraud_type(self, text: str) -> tuple[str, float]:
        """詐騙類型分類（對外介面）"""
        return self._classify_fraud_type_scored(text)

    def extract_fraud_keywords(self, text: str) -> list[str]:
        """提取所有匹配的詐騙關鍵詞"""
        matched = []
        for config in self.FRAUD_PATTERNS.values():
            for kw in config["keywords"]:
                if kw in text and kw not in matched:
                    matched.append(kw)
        return matched

    def _compute_signal_quality(
        self, transcript: TranscriptionResult
    ) -> float:
        """基於文字稿品質計算信號品質"""
        scores = []

        # 文字長度品質
        wc = transcript.word_count
        if wc < 10:
            scores.append(0.2)
        elif wc < 30:
            scores.append(0.5)
        else:
            scores.append(min(1.0, wc / 100))

        # 轉錄信心度
        scores.append(transcript.confidence)

        # 片段覆蓋率
        if transcript.segments:
            coverage = len(transcript.segments) / max(1, wc / 10)
            scores.append(min(1.0, coverage))

        return float(np.mean(scores)) if scores else 0.0

    def _build_explanation(
        self,
        fraud_type: str,
        score: float,
        keywords: list[str],
        pressure_count: int,
    ) -> str:
        """生成分析說明"""
        parts = []

        if score > 0.6:
            parts.append(f"文字內容高度符合「{fraud_type}」詐騙模式")
        elif score > 0.3:
            parts.append(f"文字內容部分符合「{fraud_type}」詐騙特徵")
        elif score > 0:
            parts.append(f"偵測到少量可疑詞彙，可能與「{fraud_type}」有關")
        else:
            parts.append("文字內容未偵測到明顯詐騙特徵")

        if keywords:
            parts.append(f"命中關鍵詞：{'、'.join(keywords[:5])}")

        if pressure_count > 0:
            parts.append(f"壓力話術計數: {pressure_count}")

        return "；".join(parts)
