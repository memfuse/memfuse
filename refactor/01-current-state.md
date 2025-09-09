# 现状评估（Current State）

基于对仓库的并行扫描与关键文件审阅，当前要点如下：

- 代码组织（Python / Poetry / FastAPI）：
  - 包名：`memfuse_core`（pyproject 声明 packages.from = src）
  - 关键模块：
    - API：`src/memfuse_core/api`（users/agents/sessions/messages/knowledge/chunks/health 等）
    - Server：`src/memfuse_core/server.py`（uvicorn factory、Hydra 配置、全局管理器初始化）
    - Services：`src/memfuse_core/services`（AppService、Memory/Buffer/Model/Connection managers、ServiceInitializer）
    - Buffer：`src/memfuse_core/buffer`（write/query/speculative/hybrid/flush_manager/config_factory）
    - Memory：`src/memfuse_core/memory`（m0/m1；未来 M2-KG 在 store/graph_* 与 rag/*）
    - Store：`src/memfuse_core/store`（pgai_store、vector_store、graph_store、keyword_store、多适配器）
    - Database：`src/memfuse_core/database`（postgres/sqlite/queued、factory、connection wrapper）
    - RAG：`src/memfuse_core/rag`（chunk/encode/retrieve/rerank/fusion；EmbeddingService 可注册到全局模型管理）
    - Utils：`src/memfuse_core/utils`（auth/validation/cache/config/perf 等）
    - Validators/ Gateway 目录存在 __pycache__ 但缺源码文件（可能已迁移或被移除，需在迁移计划中纠偏）

- 启动入口与脚本：
  - `poetry run memfuse-core` → `memfuse_core.server:main`（Hydra 读取 config）
  - `scripts/memfuse_launcher.py` 负责：Docker TimescaleDB 启动、优化、健康检查、再启动 core 服务
  - 用户偏好：`poetry run python scripts/memfuse_launcher.py`（需保持兼容）

- 配置：
  - Hydra 配置在 `config/` 下（server/buffer/memory/store/.../default.yaml + config.yaml）。
  - 代码同时存在 `utils/config_manager` 与 `global_config_manager`（有双轨迹，建议后续统一）。

- Docker / Compose：
  - `docker/compose` 下多套 compose 文件（dev/local/prod/test/pgai）。
  - 提供 TimescaleDB+pgvector/pgvectorscale、自定义 pgai。

- 测试：
  - 测试分层较全面（unit/integration/e2e/performance/contract/smoke）。
  - 合同测试覆盖 REST 资源（users/agents/sessions/messages/knowledge）。

- 文档：
  - 根 README 面向 Core Server；docs 下存在 architecture/optimization 与 _build 内大量总结文档（存在过时可能）。

初步问题清单：
- Gateway/Validators 目录空壳（仅 pycache）：
  - 可能功能实际分布在 `utils.auth`、`services` 与 API 层内部；不利于“主路径”清晰分层。
- 配置入口双轨（legacy + global）：
  - 建议统一为 `global_config_manager`，为 API/Gateway/Buffer/Memory 提供稳定读取接口。
- RAG 与 Buffer 的边界：
  - RAG 目前是独立子系统，建议在新架构中以“Buffer 插件”方式统一调度（retrieve/rerank 融入网关/缓冲管线）。
- 观测性与 Guardrail：
  - Guardrail（内容/策略）分散在 utils/validators/部分 API 内，建议提升为一等层并在 Gateway 中强制执行。

