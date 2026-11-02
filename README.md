# Bot 知识库

把她**真实读到过、查到过**的内容沉淀成可检索的知识库：读过的书段落、检索到的资料、读过的论文摘要——聊天时按话题自动检索最相关片段注入，让她能准确说出"我读到过……"，而不是凭印象编。

> 配合 [Bot 真实经历](https://github.com/Evelyn-Snow/astrbot_plugin_snow_experience) 使用：真实经历产出原文素材，本插件负责切分、向量化、检索和引用。

## 工作方式

```
真实经历完成（读书/查资料/读论文）
   ↓ 原文素材（不是读后感，是真正读到的内容）
切分（按句读边界，默认 500 字、重叠 80 字）
   ↓
向量化（复用 AstrBot 已配置的 embedding 模型，如本地 bge-m3，免费）
   ↓
存入知识库（幂等：同一来源同一内容不重复入库）
   ↓
聊天时：你的消息 → 向量检索 top-3 → 相似度过滤 → 注入上下文
```

- **只收真实经历**：不会索引你没读过的书，避免"引用没读过的内容"
- **阈值过滤**：相似度低于 0.5 的片段不注入，聊无关话题时她不会被硬塞知识
- **冷却**：同一批片段 30 分钟内不重复注入
- **工具**：她也可以主动调用 `recall_knowledge(query)` 检索

## 存储

默认在插件数据目录；可配 `index_dir` 指向任意路径（例如 Obsidian 仓库下的「知识库」文件夹）：

```
knowledge/
├─ chunks.jsonl    # 每行一个片段：来源、标题、正文、时间
├─ vectors.npy     # 向量矩阵（无 numpy 时自动改用 vectors.json）
└─ index_meta.json # 片段数、维度、最近向量化时间
```

换 embedding 模型导致维度变化时，索引自动重建（旧片段会重新向量化）。

## 配置

| 键 | 默认 | 说明 |
|---|---|---|
| min_retrieve_interval_seconds | 45 | 同会话检索最小间隔；期内复用上次结果，不重复调 embedding（每次约 300ms） |
| enabled | true | 总开关 |
| index_dir | 空 | 知识库目录；留空用插件数据目录 |
| max_chunks_per_event | 20 | 单次经历最多入库片段数 |
| chunk_size / chunk_overlap | 500 / 80 | 片段长度与重叠字数 |
| retrieve_top_k | 3 | 每轮检索片段数 |
| similarity_threshold | 0.5 | 相似度阈值（越高越严格） |
| inject_limit_chars | 600 | 注入知识的最大字数 |
| reinject_cooldown_minutes | 30 | 同批片段重复注入冷却 |
| tool_enabled | true | 提供 recall_knowledge 工具 |

## 开源方案参考

本插件采用标准 RAG 三层结构（切分 → embedding → 检索注入），实现上刻意保持轻量（无外部服务、复用宿主 embedding）：

- **LangChain `RecursiveCharacterTextSplitter`** — 按标点/段落递归切分的思路（本项目按中文句读边界实现简化版）
- **LlamaIndex** — 检索 + 引用来源的工程实践
- **Mem0** — 记忆/知识分层管理（经历 vs 事实的区分）
- **AstrBot 自带 LivingMemory** — 同为记忆引擎；本项目复用其 embedding 配置，但不共享存储（经历型知识与对话记忆语义不同）

差异点：本项目只做**个人经历型知识库**（她本人读过/查过的内容），不做通用文档 RAG；不引入 FAISS 等向量库依赖，规模在万级片段以内用矩阵点乘即可。

## 依赖

- 纯标准库 + numpy（可选）：无 numpy 时自动用 JSON 存向量并走纯 Python 相似度回退，但速度较慢。
- 需要一个 embedding 提供商（AstrBot 里配置一次即可，本地 Ollama bge-m3 免费）。
