"""
檢測報告生成器
==============
將 FusionResult 轉換為人類可讀的 Markdown、HTML 或 PDF 報告。
包含視覺化圖示、Agent 詳細分析和警示資訊。
"""
import json
import time
from pathlib import Path
from typing import Any

from src.models import FusionResult, AudioFeatures
from src.utils.logger import get_logger

logger = get_logger("ai_voice.step5.reporter")

class ReportGenerator:
    """與工程師和終端用戶溝通的報告生成器"""

    def __init__(self, output_dir: str = "reports") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, fusion: FusionResult, features: AudioFeatures, elapsed: float) -> str:
        """生成 Markdown 格式的完整報告

        Args:
            fusion: 融合後的最終判斷
            features: 原始特徵（用於顯示信號品質）
            elapsed: 耗時

        Returns:
            報告檔案路徑
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"report_{timestamp}.md"
        
        md = self._build_markdown(fusion, features, elapsed)
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(md)
            
        logger.info(f"檢測報告已生成: {report_file}")
        return str(report_file)

    def _build_markdown(self, fusion: FusionResult, features: AudioFeatures, elapsed: float) -> str:
        """構建報告內容"""
        icon = {"高風險": "🔴", "中風險": "🟡", "低風險": "🟢"}.get(fusion.risk_level, "⚪")
        
        md = f"""# AI_Voice 智慧語音詐騙檢測報告
**生成時間**: {time.strftime("%Y-%m-%d %H:%M:%S")}
**分析耗時**: {elapsed:.2f} 秒

---

## 🛡️ 總體檢測結果: {icon} {fusion.risk_level}
**詐騙機率**: {fusion.final_probability:.2%}

> [!IMPORTANT]
> **風險說明**: {self._get_risk_description(fusion)}

---

## 🤖 多 Agent 動態分析 (SE-Attention)

| Agent | 詐騙機率 | 信心度 | 動態權重 | 判定結果 |
|-------|----------|--------|----------|----------|"""

        for r in fusion.agent_results:
            w = fusion.dynamic_weights.get(r.agent_name, 0.0)
            status = "⚠️ 異常" if r.fraud_probability > 0.6 else "✅ 正常"
            name_zh = {"voiceprint": "🔊 聲紋", "semantic": "📝 語義", "memory": "🧠 記憶"}.get(r.agent_name, r.agent_name)
            md += f"\n| {name_zh} | {r.fraud_probability:.1%} | {r.confidence:.1%} | {w:.1%} | {status} |"

        md += """

---

## 🔊 聲學特徵細節 (Prosody)
| 特徵項目 | 測得數值 | 狀態描述 |
|----------|----------|----------|"""
        
        p = features.prosody
        md += f"\n| Jitter | {p.jitter:.4f} | {'過低（AI模擬跡象）' if p.jitter < 0.003 else '正常'} |"
        md += f"\n| Shimmer | {p.shimmer:.4f} | {'過低' if p.shimmer < 0.01 else '正常'} |"
        md += f"\n| HNR | {p.hnr:.1f} dB | {'異常高（非自然乾淨）' if p.hnr > 30 else '正常'} |"
        md += f"\n| F0 均值 | {p.f0_mean:.1f} Hz | - |"

        md += f"""

---

## 📜 語義分析摘要
{getattr(fusion.agent_results[1], 'explanation', '無語義分析數據')}

---

## 🧠 歷史記憶比對
{getattr(fusion.agent_results[2], 'explanation', '無記憶匹配數據')}

---

## 🛠️ 技術聲明
本報告由 **AI_Voice v3.1** 離線系統生成。
分析基於：
1. **SE-Attention 融合引擎**: 對不同信噪比環境下的 Agent 權重進行動態回歸分配。
2. **EvoAgentX 記憶框架**: 相似度匹配歷史詐騙案例。
3. **Praat 聲學分析**: 精確提取韻律特徵。

*本結果僅供參考，不具法律效應。*
"""
        return md

    def _get_risk_description(self, fusion: FusionResult) -> str:
        if fusion.risk_level == "高風險":
            return "偵測到高度典型的詐騙語義模式與 AI 合成語音特徵，建議立即掛斷並尋求官方管道驗證。"
        elif fusion.risk_level == "中風險":
            return "語音中出現部分可疑特徵，且語義與已知詐騙模式相似，請保持警覺並避免提供個人資料。"
        return "未偵測到明顯的系統性詐騙特徵，語音表現自然。"
