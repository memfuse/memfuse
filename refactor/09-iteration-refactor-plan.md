# 迭代重构计划（非兼容、低风险推进）

本计划在“不考虑向后兼容、不引入迁移模块”的前提下，采用小步快跑的迭代方式，降低一次性大改的复杂度，并以清晰的测试与验证门槛保障质量。

关键约束与约定：
- 目录与配置命名：使用 persistence（替代 store）；类/接口命名沿用 Store 后缀（如 StorePort、PgaiStore）。
- Schema 策略：db/schema/ 为唯一 DDL 来源；以 db/schema.sql 作为单一入口（entry point）一次性初始化/升级（幂等，非版本迁移）。
- Docker 不再注入/携带 schema；初始化通过工具/CI 执行 db/schema.sql。
- 启动入口与操作体验保持：poetry run python scripts/memfuse_launcher.py。

---

## 迭代 0 — 骨架与对齐（1–2 天）
目标：建立最小骨架，确保开发/CI 能并行推进。
- 代码/目录：
  - src/persistence/ 目录与 StorePort 接口占位；src/interfaces/ 端口定义占位。
  - gateway 增加 inbound/outbound filters 的注册点（空实现/示例 Filter）。
  - db/schema/ 目录与 db/schema.sql 占位（包含 include 顺序注释）。
  - config/ 分层骨架：server/gateway/guardrail/buffer/memory/persistence/database + profiles。
  - tests/ 目录骨架与各层 README、fixtures/{data,config} 规范占位。
- 文档：更新 docs/architecture/gateway.md（双向过滤）、refactor/* 对齐“persistence + Store 后缀”。
- 验证：
  - 静态检查：ruff/flake8、mypy（如启用）、markdown lint 通过。
  - 快速单测：空骨架/样例测试通过。

验收标准：
- 仓库可本地运行基本检查（lint/type/test）且 CI 绿灯。

---

## 迭代 1 — Gateway × Guardrail 双向拦截最小可用（2–4 天）
目标：实现 inbound/outbound filters 的最小链路，不改变业务算法，仅接入前后置策略框架。
- 功能：
  - inbound：鉴权、请求长度/大小、基础 PII/毒性（占位实现+配置开关）。
  - outbound：去标识化/字段删减占位；按 config/gateway/pipeline.yaml 顺序执行。
  - 统一错误模型与审计埋点框架（计数器/直方图占位）。
- 配置：
  - config/gateway/pipeline.yaml：定义 filters 顺序与开关；config/guardrail/*：阈值等参数。
- 测试：
  - unit：filters 逐一测试（启停/参数生效/错误处理）。
  - integration：API→Gateway 经由 filters 的主干流，使用 mock 的 persistence。
  - contract：OpenAPI 基本合同（路径/状态码/Schema）。
- 验证命令：
  - poetry run pytest -q tests/unit/gateway tests/integration/flows/test_gateway_pipeline.py
  - poetry run pytest -q tests/contract
  - poetry run python scripts/memfuse_launcher.py  # 健康检查

验收标准：
- filters 可按配置启停且顺序正确；主路径 integration 用例通过；合同测试通过。

---

## 迭代 2 — Buffer 插件化与 RAG 接入（3–5 天）
目标：抽象 buffer/core 接口与 pipeline；将现有能力迁入 plugins 框架（行为不重写）。
- 功能：
  - buffer/core：PluginBase、BufferPipeline、BufferPort；flush 策略与重试框架。
  - plugins：write/query/speculative/hybrid；rag 插件接入（算法不动，仅装配/入口）。
- 配置：
  - config/buffer/*.yaml：插件开关与参数；rag 的子模块参数（chunk/encode/retrieve/rerank）。
- 测试：
  - unit：PluginBase 合同测试；各插件的最小功能测试（mock persistence）。
  - integration：API→Gateway→Buffer→Memory（mock persistence）；包含 rag 路径的主场景。
- 验证命令：
  - poetry run pytest -q tests/unit/buffer tests/integration/flows/test_buffer_pipeline.py

验收标准：
- 插件化框架可独立启停，rag 插件可被 pipeline 调用；主路径 integration 通过。

---

## 迭代 3 — Memory（m0/m1）与 Persistence 最小实现（4–7 天）
目标：落地 m0/m1 的核心逻辑与最小持久化后端；完成主路径闭环。
- 功能：
  - memory/logic/m0,m1 与 policy/（时间衰减/保留占位）。
  - persistence：PgaiStore/VectorStore 的最小实现（按 StorePort）；database 连接/池化基础设施。
  - db/schema/：m0/m1 所需表/索引/触发器；db/schema.sql 包含顺序调用。
- 测试：
  - unit：m0/m1 逻辑；PgaiStore/VectorStore 行为（可引入 sqlite/内存替身）。
  - integration：API→Gateway→Buffer→Memory→Persistence 闭环；
  - e2e：通过 launcher 跑核心用例（写入→检索）。
- 验证命令：
  - psql -f db/schema.sql ...  # 或由工具触发
  - poetry run pytest -q tests/unit/memory tests/unit/persistence tests/integration/flows
  - poetry run pytest -q tests/e2e
  - poetry run python scripts/memfuse_launcher.py

验收标准：
- 主路径端到端稳定；db/schema.sql 幂等执行；关键指标基础数据可采集。

---

## 迭代 4 — 策略拓展（m2/m3/mg）与性能基线（4–8 天）
目标：拓展 Memory 高阶层与 Guardrail 策略；建立性能基线与降级策略。
- 功能：
  - memory/logic/m2,m3,mg 结构与策略占位（可条件启用）。
  - guardrail：输出合规/重写策略增强；敏感场景再校验。
- 测试：
  - unit：策略与条件启用逻辑；
  - performance：吞吐/P95/缓存命中/flush 延迟/召回-重排质量；
  - e2e：回归主路径。
- 验证命令：
  - poetry run pytest -q tests/performance -k baseline

验收标准：
- 性能基线建立并可在 CI 定时任务中稳定通过；策略启停不影响核心路径稳定。

---

## 横切保障与工程纪律
- 小 PR 原则：每次迭代内拆分为 2–4 个可审阅的 PR；每个 PR 自带最小测试与文档更新。
- 质量门槛（每 PR 必须满足）：
  - 单测/集成/合同/冒烟（按改动范围）通过；覆盖率不下降（基线 ≥80%）。
  - 关键命令成功：poetry run python scripts/memfuse_launcher.py；基本 health-check 绿。
- 幂等与数据安全：
  - db/schema.sql 需幂等；在预发/本地验证通过后再跑到共享环境。
- 文档：
  - refactor/ 与 docs/architecture/* 同步更新；新增/变更均有“为什么”的 ADR 注记。

---

## 风险与缓解
- 一次性入口 schema.sql 的升级风险：通过幂等/版本表（可选）与预发演练缓解。
- 插件化后接口变更风险：通过 StorePort/BufferPort 合同测试与 contract tests 缓解。
- 双向 filters 性能开销：提供 per-filter 开关与采样，建立性能基线观测。

---

## 验证手册（每迭代）
1) 代码静态检查：ruff/flake8、mypy（如启用）
2) 单测 → 集成 → 合同 → e2e（按范围选择执行）
3) 本地冒烟：poetry run python scripts/memfuse_launcher.py；检查 /api/v1/health
4) 如涉及 DB：执行/验证 db/schema.sql 并确认无错误与幂等
5) CI：并行跑测试金字塔；性能基线在夜间/定时任务跑



---

## 模块功能说明与开发清单（对照 10-modules-functional-spec）

### API
- 功能：纯路由 + 入参校验 + 统一错误映射；调用 Gateway。
- 开发：
  - 统一路由注册与分组；控制器内不写业务逻辑。
  - OpenAPI 校验用例与合同测试样例。
- 测试：unit（Pydantic 校验/错误码）、contract（OpenAPI）。

### Gateway
- 功能：编排主路径；注册 inbound/outbound filters；调用 Buffer/Memory；审计与指标。
- 开发：
  - request_router 与 filters/mappers；
  - inbound/outbound 注册点与执行顺序；
  - 统一错误模型、审计/指标 hook。
- 测试：unit（filters 启停/顺序）、integration（API→Gateway 主干流）。

### Guardrail
- 功能：输入/输出策略集合（毒性、PII、长度、配额、合规/脱敏等）。
- 开发：
  - 策略函数接口 `apply_inbound/apply_outbound`；
  - 配置化阈值与 per-tenant/agent 覆盖；
  - 最小可用策略占位实现（可采样）。
- 测试：unit（策略函数）、integration（随 Gateway 生效）。

### Buffer Core
- 功能：BufferPort/BufferPipeline/PluginBase；写缓冲、重试、预取、缓存。
- 开发：
  - 定义 Port/DTO；
  - Pipeline 执行模型（串/并行节点可选，占位）；
  - flush/重试策略骨架。
- 测试：unit（Port 契约）、integration（主路径含 Buffer）。

### Buffer Plugins
- 功能：write/query/speculative/hybrid；rag 作为统一插件（chunk/encode/retrieve/rerank）。
- 开发：
  - 迁移现有能力为插件实现（不改算法，改装配与入口）；
  - 插件注册与配置开关。
- 测试：unit（插件契约测试）、integration（含 rag 主场景）。

### Memory（m0/m1/m2/m3/mg + policy）
- 功能：按层的业务语义与策略（时间衰减/保留/权重）。
- 开发：
  - m0/m1 先行，m2/m3/mg 结构占位；
  - MemoryPort 的 save/retrieve/summarize/decay；
  - 严禁直连 DB，统一经 StorePort。
- 测试：unit（各层逻辑与 policy）、integration（闭环）。

### Persistence（目录名 persistence，类名沿用 Store 后缀）
- 功能：数据访问适配层；实现 StorePort；PgaiStore/VectorStore/GraphStore/KeywordStore。
- 开发：
  - StorePort 定义与最小实现（优先 PgaiStore、VectorStore）；
  - 查询/写入/批量/向量索引封装；
  - 错误与重试策略（与 Buffer 协调）。
- 测试：unit（Store 行为，sqlite/内存替身可选）、integration（闭环）。

### Database（基础设施）
- 功能：连接/池化/工厂/健康检查；不定义业务 Schema。
- 开发：
  - DatabaseFactory、pool 配置、health endpoints；
  - 只作为 Persistence 依赖。
- 测试：unit（连接/健康检查）。

### Schema（db/schema + db/schema.sql）
- 功能：唯一 DDL 来源；db/schema.sql 为单一入口并幂等。
- 开发：
  - 组织 tables/functions/triggers/views 子目录；
  - schema.sql include 顺序与注释；
  - 可选：版本表（仅记录，不做迁移）。
- 测试：本地/CI 执行校验幂等；集成/e2e 验证主路径。

### Interfaces/Models
- 功能：统一 Ports/DTO 协议；降低耦合。
- 开发：
  - Protocol/ABC；Pydantic DTO；
  - 文档化字段与校验约束。
- 测试：unit（DTO 校验、协议 mock）。

### Config（Hydra + profiles）
- 功能：分层配置 + dev/local/prod/test 叠加；global_config_manager 只读。
- 开发：
  - gateway/guardrail/buffer/memory/persistence/database/server 目录与样例；
  - profile 覆盖样例。
- 测试：unit（配置解析）、integration（配置驱动的行为变更）。

### Monitoring
- 功能：统一指标/追踪/审计。
- 开发：
  - 指标枚举与 labels 规范；
  - Gateway 埋点、Buffer 关键节点埋点。
- 测试：unit（指标上报 mock）、integration（关键指标存在）。

### Scripts
- 功能：launcher、数据检查、小型管理 CLI；可驱动 db/schema.sql。
- 开发：
  - 禁放测试脚本；
  - database_manager.py bootstrap 行为（执行 schema.sql）。
- 测试：smoke（--help/基本命令）。

### Tests
- 布局：unit / integration / contract / e2e / performance；fixtures/{data,config}
- 原则：新增插件/后端必须通过契约测试；主路径以 integration 为门槛；e2e 由 launcher 驱动；性能基线定时跑。

### Docs
- 内容：架构/目录/API/优化/ADR；
- 开发：每 PR 同步变更；关键取舍记录 ADR。
