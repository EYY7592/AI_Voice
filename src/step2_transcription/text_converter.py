"""
繁簡轉換與文本處理
==================
使用 OpenCC 將簡體中文轉換為繁體中文（台灣用語偏好）。
"""
import re

import opencc

from src.utils.logger import get_logger

logger = get_logger("ai_voice.step2.converter")


class TextConverter:
    """繁簡轉換與文字後處理器

    負責將 Whisper 輸出的簡體中文轉為繁體中文，
    同時進行文字清理與品質評估。

    Attributes:
        mode: OpenCC 轉換模式
        converter: OpenCC 轉換器實例
    """

    # 支援的 OpenCC 轉換模式
    MODES = {
        "s2t": "簡體 → 繁體",
        "s2tw": "簡體 → 繁體（台灣正體）",
        "s2twp": "簡體 → 繁體（台灣用語偏好）",
        "t2s": "繁體 → 簡體",
    }

    def __init__(self, mode: str = "s2twp") -> None:
        """初始化

        Args:
            mode: OpenCC 轉換模式。s2twp = 簡→繁+台灣用語偏好
        """
        if mode not in self.MODES:
            raise ValueError(
                f"不支援的轉換模式: {mode}。"
                f"可用: {self.MODES}"
            )
        self.mode = mode
        self.converter = opencc.OpenCC(mode)
        logger.info(f"TextConverter 初始化: {mode} ({self.MODES[mode]})")

    def to_traditional(self, text: str) -> str:
        """簡體中文 → 繁體中文

        Args:
            text: 輸入文字（可能是簡體或混合）

        Returns:
            繁體中文文字
        """
        if not text or not text.strip():
            return text

        converted = self.converter.convert(text)
        logger.debug(f"繁簡轉換: {len(text)}字 → {len(converted)}字")
        return converted

    def clean_text(self, text: str) -> str:
        """文字清理

        移除多餘空白、標準化標點、修正常見 Whisper 轉錄錯誤。

        Args:
            text: 原始文字

        Returns:
            清理後的文字
        """
        if not text:
            return text

        # 移除多餘空白
        text = re.sub(r'\s+', ' ', text).strip()

        # 統一中文標點
        replacements = {
            ',': '，',
            '.': '。',
            '?': '？',
            '!': '！',
            ':': '：',
            ';': '；',
            '(': '（',
            ')': '）',
        }
        # 只替換中文語境中的英文標點（緊鄰中文字元時）
        for eng, zht in replacements.items():
            text = re.sub(
                rf'(?<=[\u4e00-\u9fff]){re.escape(eng)}(?=[\u4e00-\u9fff])',
                zht, text
            )

        return text

    def detect_language(self, text: str) -> str:
        """偵測文字語言

        使用字元統計的簡易方法判斷語言。

        Args:
            text: 輸入文字

        Returns:
            語言代碼（"zh" / "en" / "mixed" / "unknown"）
        """
        if not text or not text.strip():
            return "unknown"

        # 清除非文字字元
        clean = re.sub(r'[\s\d\W]+', '', text)
        if not clean:
            return "unknown"

        # 統計中文字元比例
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', clean))
        total_chars = len(clean)

        ratio = chinese_chars / total_chars

        if ratio > 0.5:
            return "zh"
        elif ratio < 0.1:
            return "en"
        else:
            return "mixed"

    def process(self, text: str) -> str:
        """完整文字後處理流程：清理 → 繁簡轉換

        Args:
            text: 原始文字

        Returns:
            處理後的繁體中文文字
        """
        text = self.clean_text(text)
        text = self.to_traditional(text)
        return text
