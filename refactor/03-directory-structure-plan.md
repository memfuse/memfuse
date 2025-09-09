# 目录与模块重构方案（Directory Plan）

建议目标目录（仅规划，不立即改代码）：

```text
src/memfuse_core/
  api/                      # 控制器：纯路由 + 入参校验
  gateway/                  # 业务编排：鉴权/限流/路由/变换/审计/guardrails
    __init__.py
    request_router.py
    filters.py
    processors/
    auth/
    guardrails/             # 输入/输出/策略（与 validators 合并）
  guardrail/                # 若需要与 gateway 分离的通用策略与复用件
    __init__.py
    policies/
    validators/
  buffer/
    core/                   # 接口/基类/管线（BufferPort, BufferPipeline, PluginBase）
    plugins/
      write_buffer/
      query_buffer/
      speculative/
      hybrid/
      rag/                  # 从 rag/* 迁入（按子模块放到 plugin 内）
      code/                 # 预留扩展
  memory/
    m0/
    m1/
    m2_kg/                  # 预留
    policy/                 # 时间衰减、保留策略等
  persistence/              # 数据持久化层（pgai、vector、graph、keyword 等后端适配）
  database/                 # 维持现状
  services/                 # AppService、ServiceInitializer、Global Managers
  interfaces/               # 层间契约（Ports/DTO）
  models/                   # Pydantic 模型
  monitoring/               # 指标/日志/追踪
  utils/                    # 通用工具（保留、适当迁移 auth/validation 到 gateway/guardrail）
  server.py
```

补充：
- 新增 `db/schema/` 作为唯一的 DDL/触发器/视图定义中心；Docker 不再包含 schema 初始化脚本；本重构不引入迁移模块。


现状 → 目标的映射建议：
- `src/memfuse_core/api/*`：保留；将业务逻辑下沉至 gateway 的 orchestrator。
- `src/memfuse_core/utils/auth.py`：迁移/拆分为 `gateway/auth/*` 与中间件适配层。
- `src/memfuse_core/validators/*`：迁至 `gateway/guardrails/*`（或顶层 guardrail/ 以便多处复用）。
- `src/memfuse_core/rag/*`：整体归至 `buffer/plugins/rag/*`，以统一的 Plugin 接口接入。
- `src/memfuse_core/buffer/*`：抽象出 `buffer/core/*`，现有 write/query/speculative/hybrid 迁入 `buffer/plugins/*`。
- `src/memfuse_core/memory/*`：扩展 `policy/`，并通过 `interfaces/` 暴露 MemoryPort。

config 重构：
- 目录保留 `config/`，分层配置：
  - `config/gateway/*.yaml`（auth、ratelimit、guardrail 策略）
  - `config/buffer/*.yaml`（各插件开关/参数）
  - `config/memory/*.yaml`（策略/层开关）
  - `config/persistence/*.yaml`（pgai/vector/graph 等后端）
  - `config/server/*.yaml`（CORS、端口、reload）
  - 引入 profile：`dev/local/prod/test` 叠加覆盖。

docker 重构：
- 维持 `docker/compose/*.yml`，规范服务命名与健康检查；App 容器与 DB 容器解耦。
- 将可选后端（qdrant/neo4j）以独立 compose 覆盖提供。

测试目录重构：
- `tests/unit/<layer>/*`：api/gateway/buffer/memory/guardrail/persistence 等一层一套。
- `tests/integration/<flow>/*`：跨层主路径（e.g., messages flow, knowledge flow）。
- `tests/contract/*`：REST 资源合同与 OpenAPI 校验。
- `tests/e2e/*`：launcher → DB → core 服务端到端。

文档目录重构：
- `docs/architecture/*`：核心架构、层次职责、序列图、ADR。
- `docs/api/*`：OpenAPI/使用指南（自动生成 + 手写补充）。
- `docs/optimization/*`：性能与配置调优。

