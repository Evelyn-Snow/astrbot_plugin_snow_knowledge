"""Bot 知识库：把她真实读过的书、查到的资料、读过的论文沉淀成可检索知识库。

数据流：
  snow_experience 完成一次真实事件 → 调本插件 ingest(source, title, text)
  → 文本切分 → 用 AstrBot 的 embedding provider（本地 bge-m3）逐段向量化
  → 存 chunks.jsonl + vectors.npy（幂等，同 source+内容 hash 跳过）

聊天时：
  on_llm_request 取用户消息做 query → 检索 top-k 片段 → 注入 system_prompt
  （相似度阈值过滤、同批片段冷却、字数上限；任何失败静默跳过，不影响聊天）
另提供 recall_knowledge 工具供她主动检索。
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools
from astrbot.core.config.astrbot_config import AstrBotConfig

from .library import (
    batch_key,
    chunk_id,
    cosine_scores,
    render_injection,
    should_reinject,
    split_text,
    top_matches,
)

NAME = "astrbot_plugin_snow_knowledge"
KNOWLEDGE_MARK = "[知识库]"
META_FILE = "index_meta.json"


class SnowKnowledge(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.conf = config
        self._chunks: list[dict] | None = None
        self._vectors = None  # numpy 数组
        self._dim = 0
        self._last_batch = ""
        self._last_batch_ts = 0.0
        self._last_batch_umo = ""
        # 性能优化：同会话检索结果复用 + query 向量缓存
        self._session_hits: dict[str, tuple[float, list[dict]]] = {}
        self._query_vec_cache: dict[str, tuple[float, list[float]]] = {}

    # ---------------- 路径与配置 ----------------
    def _enabled(self) -> bool:
        return bool(self.conf.get("enabled", True))

    def _cfg(self, key: str, default):
        return self.conf.get(key, default)

    def _index_dir(self) -> Path:
        raw = str(self._cfg("index_dir", "") or "").strip()
        path = Path(os.path.expandvars(raw)) if raw else Path(str(StarTools.get_data_dir(NAME)))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _chunks_path(self) -> Path:
        return self._index_dir() / "chunks.jsonl"

    def _vectors_path(self) -> Path:
        return self._index_dir() / "vectors.npy"

    def _vectors_json_path(self) -> Path:
        return self._index_dir() / "vectors.json"

    def _meta_path(self) -> Path:
        return self._index_dir() / META_FILE

    # ---------------- 索引读写 ----------------
    def _load(self) -> None:
        if self._chunks is not None:
            return
        self._chunks = []
        try:
            with open(self._chunks_path(), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self._chunks.append(json.loads(line))
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(f"[{NAME}] 读取索引失败：{exc}")
        self._vectors = None
        self._dim = 0
        self._load_vectors()

    def _load_vectors(self) -> None:
        """优先读 npy；无 numpy 或文件缺失时读 JSON 备份。"""
        if not self._chunks:
            return
        try:
            import numpy as np

            arr = np.load(self._vectors_path())
            if arr.shape[0] == len(self._chunks):
                self._vectors = arr.astype("float32")
                self._dim = int(arr.shape[1])
                return
        except Exception:
            pass
        try:
            with open(self._vectors_json_path(), encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and len(data) == len(self._chunks):
                self._vectors = [[float(v) for v in row] for row in data]
                self._dim = len(data[0]) if data else 0
        except Exception:
            self._vectors = None

    def _save(self) -> None:
        try:
            with open(self._chunks_path(), "w", encoding="utf-8") as f:
                for chunk in self._chunks or []:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            if self._vectors is not None:
                saved = False
                try:
                    import numpy as np

                    np.save(self._vectors_path(), np.asarray(self._vectors, dtype="float32"))
                    saved = True
                except Exception:
                    saved = False
                if not saved:
                    with open(self._vectors_json_path(), "w", encoding="utf-8") as f:
                        json.dump(self._vectors, f)
            meta = {
                "count": len(self._chunks or []),
                "dim": self._dim,
                "last_embed_at": time.time(),
                "provider_id": self._embedding_provider_id(),
            }
            with open(self._meta_path(), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False)
        except Exception as exc:
            logger.warning(f"[{NAME}] 保存索引失败：{exc}")

    def _existing_ids(self) -> set[str]:
        self._load()
        return {c.get("id") for c in self._chunks or []}

    # ---------------- embedding ----------------
    def _embedding_provider(self):
        try:
            providers = self.context.get_all_embedding_providers()
        except Exception:
            return None
        for provider in providers:
            if provider is None:
                continue
            if getattr(provider, "enabled", True):
                return provider
        return providers[0] if providers else None

    def _embedding_provider_id(self) -> str:
        provider = self._embedding_provider()
        try:
            cfg = getattr(provider, "provider_config", {}) or {}
            return str(cfg.get("id") or "")
        except Exception:
            return ""

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        provider = self._embedding_provider()
        if provider is None or not texts:
            return []
        try:
            vectors = await provider.get_embeddings(texts)
            return [list(map(float, v)) for v in (vectors or [])]
        except Exception as exc:
            logger.warning(f"[{NAME}] 向量化失败：{exc}")
            return []

    # ---------------- 摄取（供 snow_experience 调用） ----------------
    async def ingest(self, source: str, title: str, text: str) -> int:
        """把一段原文素材切分入库，返回新增片段数。失败抛异常由调用方处理。"""
        if not self._enabled():
            return 0
        pieces = split_text(
            text,
            int(self._cfg("chunk_size", 500)),
            int(self._cfg("chunk_overlap", 80)),
        )
        limit = max(1, int(self._cfg("max_chunks_per_event", 20)))
        pieces = pieces[:limit]
        if not pieces:
            return 0
        known = self._existing_ids()
        new_pieces = []
        for piece in pieces:
            cid = chunk_id(source, piece)
            if cid in known:
                continue
            new_pieces.append({"id": cid, "source": str(source), "title": str(title), "text": piece, "ts": time.time()})
        if not new_pieces:
            return 0
        vectors = await self._embed([p["text"] for p in new_pieces])
        if not vectors or len(vectors) != len(new_pieces):
            raise RuntimeError("向量化失败或无可用 embedding 模型")
        dim = len(vectors[0])
        self._load()
        if self._vectors is not None and self._dim and self._dim != dim:
            logger.warning(f"[{NAME}] embedding 维度由 {self._dim} 变为 {dim}，重建索引")
            self._chunks = []
            self._vectors = None
        self._chunks = (self._chunks or []) + new_pieces
        if self._vectors is None:
            self._vectors = vectors
        else:
            try:
                import numpy as np

                block = np.asarray(vectors, dtype="float32")
                self._vectors = np.vstack([np.asarray(self._vectors, dtype="float32"), block])
            except Exception:
                base = self._vectors if isinstance(self._vectors, list) else self._vectors.tolist()
                self._vectors = list(base) + vectors
        self._dim = dim
        self._save()
        logger.info(
            f"[{NAME}] 已入库 {len(new_pieces)} 段：{title}（来源 {source}，总计 {len(self._chunks)} 段）"
        )
        return len(new_pieces)

    # ---------------- 检索 ----------------
    async def _embed_query(self, query: str) -> list[float] | None:
        """query 向量缓存（相同文本 10 分钟内复用，避免重复调 embedding）。"""
        cached = self._query_vec_cache.get(query)
        if cached and time.time() - cached[0] < 600:
            return cached[1]
        vectors = await self._embed([query])
        if not vectors:
            return None
        if len(self._query_vec_cache) > 100:
            self._query_vec_cache.clear()
        self._query_vec_cache[query] = (time.time(), vectors[0])
        return vectors[0]

    async def _retrieve(self, query: str) -> list[dict]:
        self._load()
        if not self._chunks or self._vectors is None:
            return []
        query = str(query or "").strip()
        if len(query) < 2:
            return []
        qvec = await self._embed_query(query)
        if not qvec:
            return []
        scores = cosine_scores(qvec, self._vectors)
        if not scores:
            return []
        pairs = top_matches(
            scores,
            list(range(len(scores))),
            int(self._cfg("retrieve_top_k", 3)),
            float(self._cfg("similarity_threshold", 0.5)),
        )
        hits = []
        for score, index in pairs:
            chunk = self._chunks[index]
            hits.append(
                {
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                    "text": chunk.get("text", ""),
                    "score": score,
                    "index": index,
                }
            )
        return hits

    # ---------------- 聊天注入 ----------------
    async def _session_retrieve(self, umo: str, query: str) -> list[dict]:
        """同会话最小间隔内复用上次结果，避免每轮都付 embedding 延迟。"""
        interval = max(0, int(self._cfg("min_retrieve_interval_seconds", 45)))
        now = time.time()
        cached = self._session_hits.get(umo)
        if cached and interval > 0 and now - cached[0] < interval:
            return cached[1]
        hits = await self._retrieve(query)
        if len(self._session_hits) > 100:
            self._session_hits.clear()
        self._session_hits[umo] = (now, hits)
        return hits

    @filter.on_llm_request()
    async def inject_knowledge(self, event: AstrMessageEvent, req):
        if not self._enabled():
            return
        prompt = str(getattr(req, "system_prompt", "") or "")
        if KNOWLEDGE_MARK in prompt:
            return
        query = str(getattr(event, "message_str", "") or "").strip()
        if len(query) < 2:
            return
        umo = str(event.unified_msg_origin or "")
        try:
            hits = await self._session_retrieve(umo, query)
        except Exception as exc:
            logger.debug(f"[{NAME}] 检索失败：{exc}")
            return
        if not hits:
            return
        key = batch_key([h["index"] for h in hits])
        now = time.time()
        if not should_reinject(
            self._last_batch if umo == self._last_batch_umo else "",
            key,
            self._last_batch_ts,
            now,
            int(self._cfg("reinject_cooldown_minutes", 30)),
        ):
            return
        block = render_injection(hits, int(self._cfg("inject_limit_chars", 600)))
        if not block:
            return
        self._last_batch, self._last_batch_ts, self._last_batch_umo = key, now, umo
        req.system_prompt = prompt + "\n\n" + block

    # ---------------- 工具 ----------------
    @filter.llm_tool(name="recall_knowledge")
    async def recall_knowledge(self, event: AstrMessageEvent, query: str) -> str:
        """检索你自己的知识库：你真实读过的书、查过的资料、读过的论文。

        当有人问起你读到过什么、某本书讲了什么、之前查过的某个话题时调用。

        Args:
            query(string): 要检索的话题或问题
        """
        if not self._enabled() or not bool(self._cfg("tool_enabled", True)):
            return "知识库当前不可用。"
        try:
            hits = await self._retrieve(query)
        except Exception as exc:
            return f"检索知识库失败：{exc}"
        if not hits:
            return f"知识库里没有和「{query}」相关的内容。"
        lines = [f"和「{query}」相关的知识片段："]
        for hit in hits:
            snippet = str(hit["text"])[:200]
            lines.append(f"- 《{hit['title']}》：{snippet}（相似度 {hit['score']:.2f}）")
        return "\n".join(lines)

    # ---------------- 辅助 ----------------
    def stats(self) -> dict:
        self._load()
        return {"count": len(self._chunks or []), "dim": self._dim}
