# 设计提案（根据新增需求与合并分支）

本提案结合 feat/121-metadata 分支已合并的网关实现（gateway/api_gateway.py、metadata_router.py、processors.py、validators/guardrails.py 等），在不违背行业惯例的前提下，对以下 6 点做出统一设计与落地规划。

本轮重构说明：
- 不考虑兼容现有实现；以最优架构、可扩展、模块化为主要目标。
- 不引入迁移模块；集中 Schema 于 db/schema/，由工具一次性引导（bootstrap）。


## 0. 背景与现状校准
- 已合并 feat/121-metadata：引入了 gateway 层与 validators/guardrails，测试/文档亦包含网关相关用例与说明（docs/architecture/gateway.md）。
- 需避免命名冲突：在非 AI 模型上下文中不使用“transformer”术语。
- 需将 Memory 的逻辑与数据库访问/Schema 完全解耦，并集中定义 schema。
- docker 仅做容器与编排；tests 需重塑为可扩展且自解释的结构。

---

## 1) Gateway 与 Guardrail 的双向（入/出）管道设计

目标：在“业界常见 Gateway 负责编排/路由/接入、Guardrail 负责策略化校验”的边界内，提供一个“双向拦截（inbound/outbound）”能力统一点，最大化融合重叠功能，最小化概念混淆。

- 概念划分（行业一致）：
  - Gateway（编排/接入层）：鉴权、限流、请求/响应归一化、路由到下游（Buffer/Memory），统一错误模型与审计。
  - Guardrail（策略层）：配置化的输入/输出策略（毒性/PII/长度/配额/合规/结果过滤/重写等），不持有业务状态，尽量无副作用。
- 双向接口（Interceptors）
  - inbound_filters：在请求进入 Gateway 后、到达业务管线前执行；典型校验：鉴权/配额/请求大小/毒性/PII 等。
  - outbound_filters：在业务响应返回 API 层前执行；典型处理：去标识化、敏感字段删减、输出风格/合规约束。
  - 两类过滤器均通过 Guardrail 策略集合统一实例化与挂载，由 Gateway 调用（即“由 Gateway 统一驱动 Guardrail 的 pre/post 执行”）。
- 与现有代码的映射
  - gateway/processors.py：保留“编排器”定位，命名统一为 processors/filters（见术语规范）。
  - validators/guardrails.py：迁移/拆分为 guardrail 策略/校验器集合；Gateway 通过 inbound/outbound 注册点调用。
- 配置
  - config/gateway/pipeline.yaml：定义 inbound_filters 与 outbound_filters 的顺序、开关与参数（per-tenant/agent 可覆写）。
  - config/guardrail/*.yaml：策略参数（如阈值、敏感词表、PII 模式、长度/token 限制、重写/过滤策略等）。

---

## 2) 术语规范：避免“transformer”冲突
- 文件与模块
  - gateway/filters.py（或 mappers.py）：用于请求/响应归一化/映射。
  - gateway/processors/：业务前后处理器集合（非模型“Transformer”）。
- 文档
  - 文档与注释统一使用 filter/mapper/interceptor，而非 transformer。

---

## 3) Memory 逻辑与数据库解耦 + 统一 Schema 中心

目标：将“记忆层的业务逻辑（m0/m1/m2/m3/mg）”与“存储/Schema/迁移”彻底分离，并在一个统一位置定义与演进所有数据库 Schema，Docker 不再存放或复制 Schema。

- 目录与职责
  - memory/logic/
    - m0/ m1/ m2/ m3/ mg/：纯业务逻辑（事实/事件/语义/程序/图谱），不可直接访问 DB，仅通过 Ports。
    - policy/：时间衰减、保留/淘汰、权重策略等。
  - memory/ports/（或 interfaces/）
    - MemoryPort：save/retrieve/summarize/decay 等统一接口，供 Gateway/Buffer 调用。
  - persistence/ 与 database/
    - persistence/*：具体后端实现（pgai/vector/graph/keyword…），实现 StorePort（接口命名保留 Store 后缀，目录为 persistence），仅处理数据访问。
    - database/*：连接、工厂、队列等基础 DB 设施，不含业务 schema 定义。
- 统一 Schema 位置与版本化
  - 新建 db/schema/ 作为唯一 Schema 源：
    - db/schema/tables/*.sql
    - db/schema/functions/*.sql
    - db/schema/triggers/*.sql
    - db/schema/views/*.sql
    - 不引入迁移模块；提供 db/schema.sql 作为单一入口（按顺序包含上述 DDL），由工具一次性初始化。
    - scripts/database_manager.py 仅调用迁移层，不再散落执行 SQL；Docker 不再附带 init-scripts。
  - 扩展性与后向兼容
    - Schema 设计遵循命名/作用域（tenant/agent/session）规范；保留保守的可空/默认策略。
    - Mx 层的表/索引/触发器集中管理，支持条件编译/分支（如开启/关闭 m2/m3/mg）。

---

## 4) config 目录的分层与结构
- 顶层结构（示例）
  - config/
    - server/ (default.yaml, dev.yaml, prod.yaml, test.yaml)
    - gateway/ (pipeline.yaml, auth.yaml, ratelimit.yaml)
    - guardrail/ (toxicity.yaml, pii.yaml, quota.yaml, output.yaml)
    - buffer/ (write.yaml, query.yaml, speculative.yaml, rag.yaml)
    - memory/ (m0.yaml, m1.yaml, m2.yaml, m3.yaml, mg.yaml, policy.yaml)
    - persistence/ (pgai.yaml, vector.yaml, graph.yaml, keyword.yaml)
    - database/ (postgres.yaml, sqlite.yaml, pool.yaml)
  - 管理方式
    - 以 global_config_manager 为统一只读入口，Hydra 负责 profile 叠加（dev/local/prod/test）。
    - gateway 读取 pipeline 配置挂载 inbound/outbound filters；guardrail 策略按 scope 覆写。

---

## 5) docker 边界与清理
- docker/ 仅包含：Dockerfile、compose、启动/部署脚本、健康检查。
- 移除/迁移所有 schema/SQL 到 db/schema/；compose 启动时不再注入 schema。
- database 初始化/升级统一通过 scripts/database_manager.py（或 CLI）触发，不再由 docker init-scripts 驱动。
- 引入可选后端（qdrant/neo4j）以覆盖 compose 提供，保持容器边界清晰。

---

## 6) tests 目标结构与文档/数据编排

目标：统一、可扩展、自解释的测试布局；每个层/流均有 README 与样例；小规模数据/配置集中管理，便于复用。

- 目录布局（建议）
  - tests/
    - unit/
      - api/ gateway/ buffer/ memory/ guardrail/ persistence/ database/ utils/
      - README.md：约定与命名规则、mock/fixture 规范
    - integration/
      - flows/（messages_flow.py, knowledge_flow.py 等）
      - adapters/（最小后端替身，如 sqlite、内存型 vector）
      - README.md：如何组合出“主路径”测试
    - contract/
      - OpenAPI/资源契约校验
      - README.md：契约来源、如何更新
    - e2e/
      - 以 scripts/memfuse_launcher.py 驱动端到端用例
      - README.md：如何本地/CI 运行
    - performance/
      - KPI 基线（吞吐、P95 延迟、缓存命中、flush 延迟、召回/重排质量）
      - README.md：基线维护与阈值设定
    - fixtures/
      - config/ 小配置片段（Hydra overrides）
      - data/ 小规模数据集（≤ 1–2MB/文件），命名规范与 LICENSE 说明
      - README.md：如何制作/扩充小数据集
- 运行策略
  - CI 中分层并行执行（unit → integration → contract → e2e），失败快速反馈；性能基线定期跑。
  - 覆盖率阈值从 80% 起步，逐步提升；关键路径以 contract/integration 为门槛。

---

## 渐进落地（与当前分支对齐）
- 保留现有 gateway/* 代码，按本提案引入 inbound/outbound filters 的注册点与配置；将 validators/guardrails 作为策略集合由 Gateway 调用。
- 在不更改算法/行为的前提下，将 rag/* 作为 buffer/plugins/rag/* 接入，统一 plugin 接口；旧入口保留过渡期适配。
- 引入 db/schema/ + db/schema.sql；将 docker 下 schema 脚本迁出；scripts/database_manager.py 仅负责一次性初始化/升级引导。
- config 逐步拆分到 gateway/guardrail/buffer/memory/persistence/database/server；以 dev/local/prod/test profile 管理。
- tests 先补齐各层 README 与 fixtures/data 规范，再按层/流补充与归并用例。

---

## 决策点（请确认）
1) Gateway 统一驱动 Guardrail 的双向过滤（inbound/outbound）是否接受？如是，将在 gateway 配置中增加双向注册点。
2) Schema 集中在 db/schema/，不引入迁移模块；采用一次性入口 db/schema.sql（由工具触发执行），Docker 不再承载 schema。
3) Memory 逻辑（m0/m1/m2/m3/mg）与 Persistence/Database 完全解耦，统一通过 Ports 访问，是否接受？
4) 术语统一替换为 filter/mapper/interceptor，避免 transformer 冲突，是否接受？
5) tests 的布局与 README/fixtures/data 规范是否满足团队预期？需不需要额外的生成脚本？

