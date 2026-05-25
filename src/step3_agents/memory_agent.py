"""
記憶學習系統 Agent（Agent C）
==============================
EvoAgentX 風格三層記憶架構：
  - WorkingMemory：會話層級短期記憶
  - PersistentMemory：FAISS 向量索引長期記憶
  - MemoryOptimizer：去重 + 時間衰減

使用 Sentence-BERT 進行語義嵌入，FAISS 進行高效相似度搜尋。
"""
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.step3_agents.base_agent import BaseAgent
from src.models import TranscriptionResult, AgentResult, MemoryMatch
from src.privacy import build_private_case_metadata, normalize_persistent_metadata
from src.utils.logger import get_logger
from src.utils.exceptions import ModelLoadError, AgentAnalysisError

logger = get_logger("ai_voice.step3.memory")


# ============================================================
# 三層記憶子系統
# ============================================================

class WorkingMemory:
    """短期記憶（會話層級）

    儲存當前會話的分析結果，用於上下文感知。
    容量有限，超過後自動淘汰最舊的記錄。
    """

    def __init__(self, capacity: int = 50) -> None:
        """初始化

        Args:
            capacity: 最大記錄數量
        """
        self.capacity = capacity
        self._store: dict[str, Any] = {}
        self._history: list[dict] = []

    def store(self, key: str, value: Any) -> None:
        """儲存鍵值對"""
        self._store[key] = value
        # 容量控制
        if len(self._store) > self.capacity:
            oldest = list(self._store.keys())[0]
            del self._store[oldest]

    def retrieve(self, key: str) -> Any:
        """取得指定鍵的值"""
        return self._store.get(key)

    def get_recent_judgments(self, n: int = 10) -> list[dict]:
        """取得最近 n 筆判決記錄

        Args:
            n: 返回數量

        Returns:
            最近的判決記錄清單
        """
        return self._history[-n:]

    def add_judgment(self, judgment: dict) -> None:
        """新增判決記錄到歷史"""
        judgment["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._history.append(judgment)
        if len(self._history) > self.capacity:
            self._history.pop(0)

    def clear(self) -> None:
        """清空短期記憶"""
        self._store.clear()
        self._history.clear()


class PersistentMemory:
    """長期記憶（FAISS 向量索引）

    使用 FAISS 進行高效向量相似度搜尋，
    每個案例包含語義嵌入向量和元資料。
    """

    def __init__(
        self,
        index_path: str | None = None,
        meta_path: str | None = None,
        embedding_dim: int = 384  # all-MiniLM-L6-v2 的嵌入維度
    ) -> None:
        """初始化

        Args:
            index_path: FAISS 索引檔案路徑
            meta_path: 案例元資料 JSON 路徑
            embedding_dim: 嵌入向量維度
        """
        self.index_path = index_path
        self.meta_path = meta_path
        self.embedding_dim = embedding_dim
        self._index = None
        self._metadata: list[dict] = []

    def load(self) -> bool:
        """載入索引和元資料

        Returns:
            是否成功載入
        """
        try:
            import faiss
        except ImportError:
            logger.warning("faiss 未安裝，記憶系統不可用")
            return False

        loaded = False

        # 嘗試載入現有索引
        if self.index_path and Path(self.index_path).exists():
            try:
                self._index = faiss.read_index(self.index_path)
                loaded = True
                logger.info(f"FAISS 索引載入成功: {self._index.ntotal} 個向量")
            except Exception as e:
                logger.warning(f"FAISS 索引載入失敗: {e}")

        # 建立空索引
        if self._index is None:
            self._index = faiss.IndexFlatIP(self.embedding_dim)  # 內積（餘弦相似度）
            logger.info(f"建立新的 FAISS 索引（維度 {self.embedding_dim}）")

        # 載入元資料
        if self.meta_path and Path(self.meta_path).exists():
            try:
                with open(self.meta_path, "r", encoding="utf-8") as f:
                    self._metadata = json.load(f)
                
                # 確保所有案例都有 ID
                import uuid
                for i, meta in enumerate(self._metadata):
                    self._metadata[i] = normalize_persistent_metadata(meta)
                    meta = self._metadata[i]
                    if "id" not in meta:
                        meta["id"] = str(uuid.uuid4())
                        
                logger.info(f"元資料載入成功: {len(self._metadata)} 筆案例")
                loaded = True
            except Exception as e:
                logger.warning(f"元資料載入失敗: {e}")

        return loaded

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> list[MemoryMatch]:
        """搜尋最相似的歷史案例

        Args:
            query_embedding: 查詢向量（嵌入維度）
            top_k: 返回前 K 個結果

        Returns:
            MemoryMatch 清單（按相似度降序）
        """
        if self._index is None or self._index.ntotal == 0:
            return []

        # 正規化查詢向量（用於餘弦相似度）
        query = query_embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(query)
        if norm > 0:
            query = query / norm

        # FAISS 搜尋
        k = min(max(top_k * 3, top_k + 10), self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx] if idx < len(self._metadata) else {}
            if meta.get("_deleted"):
                continue
            results.append(MemoryMatch(
                similarity=float(max(0.0, min(1.0, score))),
                case_text="",
                fraud_type=meta.get("fraud_type", "未知"),
                timestamp=meta.get("timestamp", ""),
            ))
            # 悄悄把 ID 也塞給前端用的字典中，雖然 dataclass 沒有，但可以用在後續包裝
            results[-1].case_id = meta.get("id", "")
            if len(results) >= top_k:
                break

        return results

    def insert(self, embedding: np.ndarray, metadata: dict) -> None:
        """插入新案例至索引

        Args:
            embedding: 嵌入向量
            metadata: 案例元資料
        """
        if self._index is None:
            return

        # 正規化
        vec = embedding.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        self._index.add(vec)
        
        import uuid
        metadata["id"] = metadata.get("id", str(uuid.uuid4()))
        metadata["timestamp"] = metadata.get(
            "timestamp", time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self._metadata.append(metadata)

        logger.debug(f"記憶庫新增案例，目前共 {self._index.ntotal} 筆")

    def save(self) -> None:
        """儲存索引和元資料至檔案"""
        if self._index is not None and self.index_path:
            import faiss
            Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, self.index_path)
            logger.info(f"FAISS 索引已儲存: {self.index_path}")

        if self._metadata and self.meta_path:
            Path(self.meta_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
            logger.info(f"元資料已儲存: {self.meta_path}")

    @property
    def total_cases(self) -> int:
        """記憶庫中的案例總數"""
        return self._index.ntotal if self._index else 0


class MemoryOptimizer:
    """記憶優化器

    負責記憶庫的維護：去重和時間衰減。
    """

    def __init__(self, persistent_memory: PersistentMemory) -> None:
        self.memory = persistent_memory

    def deduplicate(self, threshold: float = 0.95) -> int:
        """去除相似度超過閾值的重複案例

        Args:
            threshold: 相似度閾值（0~1），大於此值視為重複

        Returns:
            刪除的重複案例數量
        """
        # 簡化版：標記重複但不實際刪除（FAISS 不支援刪除）
        if self.memory._index is None or self.memory.total_cases < 2:
            return 0

        duplicates = 0
        for i, meta in enumerate(self.memory._metadata):
            if meta.get("_deleted"):
                continue
            # 檢查後續項目是否重複
            for j in range(i + 1, len(self.memory._metadata)):
                if self.memory._metadata[j].get("_deleted"):
                    continue
                # 比較文字相似度（簡單版）
                text_i = meta.get("text", "")
                text_j = self.memory._metadata[j].get("text", "")
                if text_i and text_j and text_i == text_j:
                    self.memory._metadata[j]["_deleted"] = True
                    duplicates += 1

        logger.info(f"去重完成: 標記 {duplicates} 筆重複")
        return duplicates

    def apply_time_decay(self, decay_rate: float = 0.01) -> None:
        """對歷史案例套用時間衰減加權

        Args:
            decay_rate: 每天的衰減率
        """
        current_time = time.time()
        for meta in self.memory._metadata:
            ts = meta.get("timestamp", "")
            if ts:
                try:
                    case_time = time.mktime(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))
                    days_old = (current_time - case_time) / 86400
                    meta["decay_weight"] = max(0.1, 1.0 - decay_rate * days_old)
                except (ValueError, OverflowError):
                    meta["decay_weight"] = 0.5

        logger.info("時間衰減加權完成")


# ============================================================
# Memory Agent 主類別
# ============================================================

class MemoryAgent(BaseAgent):
    """Agent C：EvoAgentX 風格記憶學習系統

    使用 Sentence-BERT 生成語義嵌入，
    FAISS 索引搜尋歷史詐騙案例，
    三層記憶架構實現會話感知和長期學習。
    """

    def __init__(
        self,
        index_path: str | None = None,
        meta_path: str | None = None,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """初始化

        Args:
            index_path: FAISS 索引路徑
            meta_path: 元資料路徑
            embedding_model: Sentence-BERT 模型名稱
        """
        super().__init__(name="memory")
        self.embedding_model_name = embedding_model
        self._embedding_model = None

        # 三層記憶
        self.working_memory = WorkingMemory(capacity=50)
        self.persistent_memory = PersistentMemory(
            index_path=index_path,
            meta_path=meta_path,
        )
        self.optimizer = MemoryOptimizer(self.persistent_memory)

    def load_model(self) -> None:
        """載入嵌入模型和記憶索引"""
        # 載入 FAISS 索引
        self.persistent_memory.load()

        # 載入 Sentence-BERT 嵌入模型
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"載入嵌入模型: {self.embedding_model_name}...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
            self.is_loaded = True
            logger.info("嵌入模型載入完成")
        except ImportError:
            logger.warning("sentence-transformers 未安裝，使用隨機嵌入")
        except Exception as e:
            logger.warning(f"嵌入模型載入失敗: {e}")

    def analyze(self, transcript: TranscriptionResult) -> AgentResult:
        """比對記憶庫中的歷史案例

        Args:
            transcript: Step2 轉錄結果

        Returns:
            AgentResult（Top-K 匹配案例 + 最終分數）
        """
        try:
            text = transcript.text

            if not text or not text.strip():
                return self._create_result(
                    fraud_probability=0.0,
                    confidence=0.0,
                    signal_quality=0.0,
                    details={"error": "文字稿為空"},
                    explanation="無文字可供記憶比對",
                )

            # 生成查詢嵌入
            query_embedding = self._encode(text)

            # 搜尋歷史案例
            matches = self.persistent_memory.search(query_embedding, top_k=5)

            # 計算詐騙機率
            if matches:
                # 加權平均（相似度越高，權重越大）
                weighted_sum = sum(m.similarity ** 2 for m in matches)
                total_weight = sum(m.similarity for m in matches)
                fraud_prob = weighted_sum / max(total_weight, 1e-6)

                # 信心度（基於最佳匹配的相似度）
                confidence = matches[0].similarity

                # 詳細資訊
                details = {
                    "top_match_similarity": matches[0].similarity,
                    "matched_cases": len(matches),
                    "matches": [
                        {
                            "similarity": m.similarity,
                            "fraud_type": m.fraud_type,
                            "text_preview": m.case_text[:50],
                        }
                        for m in matches[:3]
                    ],
                }

                explanation = self._build_explanation(matches)

            else:
                fraud_prob = 0.0
                confidence = 0.0
                details = {
                    "top_match_similarity": 0.0,
                    "matched_cases": 0,
                    "memory_size": self.persistent_memory.total_cases,
                }
                explanation = (
                    f"記憶庫中無匹配案例"
                    f"（共 {self.persistent_memory.total_cases} 筆）"
                )

            # 記錄到短期記憶
            self.working_memory.add_judgment({
                "text_preview": text[:50],
                "fraud_probability": fraud_prob,
                "matched_cases": len(matches),
            })

            signal_quality = self._compute_signal_quality(
                transcript, matches
            )

            return self._create_result(
                fraud_probability=fraud_prob,
                confidence=confidence,
                signal_quality=signal_quality,
                details=details,
                explanation=explanation,
            )

        except Exception as e:
            raise AgentAnalysisError(f"記憶系統分析失敗: {e}") from e

    def _encode(self, text: str) -> np.ndarray:
        """將文字轉換為嵌入向量

        Args:
            text: 輸入文字

        Returns:
            嵌入向量（384 維）
        """
        if self._embedding_model is not None:
            embedding = self._embedding_model.encode(
                text, normalize_embeddings=True
            )
            return np.array(embedding, dtype=np.float32)
        else:
            # 隨機嵌入（模型未載入時的 fallback）
            rng = np.random.RandomState(hash(text) % 2**31)
            vec = rng.randn(self.persistent_memory.embedding_dim).astype(np.float32)
            return vec / np.linalg.norm(vec)

    def store_episode(
        self,
        text: str,
        fraud_type: str,
        confirmed: bool = True
    ) -> None:
        """將確認的案例寫入長期記憶

        Args:
            text: 案例文字
            fraud_type: 詐騙類型
            confirmed: 是否已確認為詐騙
        """
        embedding = self._encode(text)
        self.persistent_memory.insert(
            embedding,
            build_private_case_metadata("", text, fraud_type, confirmed),
        )
        logger.info(f"案例已寫入記憶庫: {fraud_type}, 共 {self.persistent_memory.total_cases} 筆")

    def delete_case(self, case_id: str) -> bool:
        """從長期記憶中刪除一筆案例。

        FAISS IndexFlatIP does not support removing one vector safely. Use a
        tombstone so searches and history skip the case without needing raw
        transcript text to rebuild the index.
        """
        memory = self.persistent_memory
        idx_to_del = -1
        for i, meta in enumerate(memory._metadata):
            if meta.get("id") == case_id:
                idx_to_del = i
                break
        
        if idx_to_del == -1:
            return False
            
        memory._metadata[idx_to_del]["_deleted"] = True
        memory._metadata[idx_to_del]["deleted_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"從記憶庫隱藏案例: {memory._metadata[idx_to_del].get('fraud_type')} - {case_id}")
        memory.save()
        return True

    def optimize(self) -> dict:
        """執行記憶優化

        Returns:
            {"deduplicated": int, "decayed": int}
        """
        dedup_count = self.optimizer.deduplicate()
        self.optimizer.apply_time_decay()
        return {"deduplicated": dedup_count, "decayed": self.persistent_memory.total_cases}

    def save(self) -> None:
        """儲存記憶庫"""
        self.persistent_memory.save()

    def _compute_signal_quality(
        self,
        transcript: TranscriptionResult,
        matches: list[MemoryMatch],
    ) -> float:
        """計算信號品質"""
        scores = []

        # 文字稿品質
        wc = transcript.word_count
        scores.append(min(1.0, wc / 50))

        # 記憶庫覆蓋度
        if self.persistent_memory.total_cases > 0:
            scores.append(min(1.0, self.persistent_memory.total_cases / 100))
        else:
            scores.append(0.1)

        # 匹配品質
        if matches:
            scores.append(matches[0].similarity)

        return float(np.mean(scores))

    def _build_explanation(self, matches: list[MemoryMatch]) -> str:
        """生成分析說明"""
        if not matches:
            return "記憶庫中無匹配案例"

        top = matches[0]
        parts = [
            f"記憶庫找到 {len(matches)} 個相似案例",
            f"最佳匹配相似度 {top.similarity:.1%}",
            f"類型: {top.fraud_type}",
        ]

        if top.case_text:
            preview = top.case_text[:30] + "..." if len(top.case_text) > 30 else top.case_text
            parts.append(f"案例摘要:「{preview}」")

        return "；".join(parts)
