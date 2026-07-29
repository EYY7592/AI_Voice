"""可解釋的內容防詐分析核心。"""
from __future__ import annotations

from dataclasses import dataclass, field
import re


FRAUD_CATEGORIES = (
    "假投資", "假客服／解除分期", "假網拍／購物廣告", "假冒機構／檢警",
    "冒充親友／交友", "貸款／信用", "釣魚／帳號驗證", "中獎／獎金詐騙",
)
NORMAL_TYPE = "正常／未知"


@dataclass(frozen=True)
class Message:
    text: str
    timestamp: str | None = None
    speaker: str | None = None


@dataclass(frozen=True)
class ScamAnalysis:
    risk_score: int | None
    risk_level: str
    categories: list[str]
    evidence: list[dict[str, object]] = field(default_factory=list)
    safety_actions: list[str] = field(default_factory=list)
    status: str = "analyzed"
    bert_probability: float | None = None
    window_count: int = 0

    @property
    def fraud_probability(self) -> float:
        """相容既有離線評估工具；產品 API 不公開為機率。"""
        return (self.risk_score or 0) / 100


class ScamAnalyzer:
    """以透明話術證據為主、BERT 為輔的分析器。"""

    def __init__(self, *, medium_risk_score: int = 40, high_risk_score: int = 70) -> None:
        if not 0 <= medium_risk_score <= high_risk_score <= 100:
            raise ValueError("風險門檻必須介於 0 到 100，且中風險不得高於高風險。")
        self.medium_risk_score = medium_risk_score
        self.high_risk_score = high_risk_score

    CATEGORY_SIGNALS = {
        "假投資": ("投資", "高報酬", "保證獲利", "穩賺", "老師帶單", "內部消息", "虛擬貨幣", "入金", "出金", "私密群"),
        "假客服／解除分期": ("客服", "解除分期", "重複扣款", "退款", "訂單異常", "驗證碼", "誤設會員", "操作ATM"),
        "假網拍／購物廣告": ("賣貨便", "網拍", "貨到付款", "私下交易", "下單連結", "超商代碼", "限量搶購", "匯款後出貨"),
        "假冒機構／檢警": ("檢察官", "警察局", "法院", "涉嫌洗錢", "監管帳戶", "安全帳戶", "通緝", "偵查不公開"),
        "冒充親友／交友": ("換電話", "幫我匯款", "借我錢", "急用", "交友", "愛你", "見面前", "代付", "保釋金"),
        "貸款／信用": ("貸款", "代辦貸款", "信用瑕疵", "保證過件", "美化帳戶", "預付手續費", "刷流水", "提高額度"),
        "釣魚／帳號驗證": ("點擊連結", "帳號驗證", "帳戶停用", "重新登入", "驗證碼", "密碼", "身分驗證", "釣魚"),
        "中獎／獎金詐騙": ("中獎", "獎金", "保證金", "領取獎金"),
    }
    PHRASE_SIGNALS = {
        "要求付款": re.compile(r"(立即|馬上|現在|限時).{0,12}(匯款|轉帳|付款|入金)"),
        "指定帳戶": re.compile(r"(匯|轉|存).{0,12}(指定|安全|監管).{0,6}帳戶"),
        "阻止查證": re.compile(r"(不要|不可|禁止).{0,8}(告訴|聯絡|查證|報警)"),
        "索取機密": re.compile(r"(提供|告知|輸入).{0,8}(驗證碼|密碼|帳號|卡號)"),
        "付費領獎": re.compile(r"(中獎|獎金).{0,12}(支付|匯款|保證金|手續費)"),
        "異常通知索資": re.compile(r"(客服|平台).{0,12}(帳戶|訂單).{0,6}(異常|停用).{0,12}(依指示|提供資料|驗證)"),
        "保證獲利招攬": re.compile(r"(穩賺|保證獲利|保證回本).{0,12}(投資|方案|加入|入金|獲利)"),
    }
    PRESSURE_SIGNALS = ("立即", "馬上", "限時", "最後機會", "否則", "保密")
    ACTIONS = {
        "假投資": "停止入金或下載不明投資 App，改由合法金融機構查證。",
        "假客服／解除分期": "停止操作網銀或 ATM，改用官方網站上的客服管道查證。",
        "假網拍／購物廣告": "不要私下匯款或點擊陌生下單連結，使用平台內建交易機制。",
        "假冒機構／檢警": "不要依來電指示轉帳；檢警不會要求匯入安全或監管帳戶。",
        "冒充親友／交友": "改用原本已知的聯絡方式向本人確認，不要代付或匯款。",
        "貸款／信用": "不要先付手續費或提供網銀資料，向合法金融機構查證。",
        "釣魚／帳號驗證": "不要點擊連結或提供密碼與驗證碼，直接開啟官方網站或 App。",
        "中獎／獎金詐騙": "不要先支付保證金或手續費；改由主辦單位官方管道查證。",
    }
    DEFAULT_ACTIONS = (
        "先停止付款、點擊連結或提供驗證碼。",
        "改用自己查到的官方管道查證；有疑慮可聯絡 165 反詐騙諮詢專線。",
    )

    def analyze_text(self, text: str, *, bert_probability: float | None = None, bert_risk_score: int | None = None, window_count: int = 0) -> ScamAnalysis:
        clean = text.strip()
        meaningful = re.sub(r"\s|[^\w\u4e00-\u9fff]", "", clean)
        if len(meaningful) < 6:
            return ScamAnalysis(
                None,
                "資料不足",
                [NORMAL_TYPE],
                safety_actions=list(self.DEFAULT_ACTIONS),
                status="insufficient_data",
            )

        evidence: list[dict[str, object]] = []
        categories: list[str] = []
        points = 0
        for category, signals in self.CATEGORY_SIGNALS.items():
            matched = [signal for signal in signals if signal in clean]
            if matched:
                categories.append(category)
                evidence.extend({"category": category, "text": signal, "kind": "話術"} for signal in matched)
                points += min(36, 12 * len(matched))

        for label, pattern in self.PHRASE_SIGNALS.items():
            for match in pattern.finditer(clean):
                evidence.append({"category": label, "text": match.group(0), "kind": "行為模式"})
                points += 25

        pressure = [signal for signal in self.PRESSURE_SIGNALS if signal in clean]
        evidence.extend({"category": "施壓話術", "text": signal, "kind": "壓力"} for signal in pressure)
        points += min(16, 8 * len(pressure))
        strong_pattern_score = (
            self.high_risk_score if any(item["kind"] == "行為模式" for item in evidence) else 0
        )
        rule_score = min(100, max(points, strong_pattern_score))
        model_score = max(0, min(90, bert_risk_score or 0))
        agreement = 10 if rule_score >= self.medium_risk_score and model_score >= self.medium_risk_score else 0
        score = min(100, max(rule_score, model_score) + agreement)
        categories = categories or [NORMAL_TYPE]
        actions = [self.ACTIONS[item] for item in categories if item in self.ACTIONS]
        actions.extend(action for action in self.DEFAULT_ACTIONS if action not in actions)
        return ScamAnalysis(score, self._risk_level(score), categories, evidence, actions, bert_probability=bert_probability, window_count=window_count)

    def analyze_messages(self, messages: list[Message], *, source_type: str = "plain_text", bert_probability: float | None = None) -> ScamAnalysis:
        del source_type
        return self.analyze_text("\n".join(message.text for message in messages), bert_probability=bert_probability)

    def _risk_level(self, score: int) -> str:
        if score >= self.high_risk_score:
            return "高風險"
        if score >= self.medium_risk_score:
            return "中風險"
        return "低風險"
