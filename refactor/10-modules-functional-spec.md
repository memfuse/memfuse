# 模块功能说明与接口契约（Functional Spec by Module)

本文定义各模块的功能职责、对外接口（Ports/DTO）、与配置位置，统一命名及非目标，作为开发与评审依据。

---

## 1. API（FastAPI Controllers）
- 职责：
  - 路由与入参校验（Pydantic）；错误到统一异常模型的映射；调用 Gateway。
- 对外接口：HTTP（REST/OpenAPI）。
- 依赖：Gateway。
- 配置：config/server/*（端口/CORS/reload）。
- 非目标：不包含业务编排与策略判断。

## 2. Gateway（Orchestrator）
- 职责：
  - 鉴权、限流、请求/响应归一化（filters/mappers）；主流程编排；审计/监控埋点。
  - 双向拦截：inbound_filters（输入方向）/ outbound_filters（输出方向）；统一驱动 Guardrail 策略集合。
- 对外接口：
  - 调用下游 Buffer/Memory；暴露编排接口 `Gateway.handle(request) -> response`。
- 依赖：Guardrail、Buffer、Memory（按路由选择）。
- 配置：config/gateway/pipeline.yaml（filters 顺序/开关）、config/gateway/{auth,ratelimit}.yaml。
- 非目标：不承载持久化逻辑。

## 3. Guardrail（策略层）
- 职责：
  - 策略化输入/输出检查（毒性、PII、长度、配额、合规、脱敏等），纯函数或幂等；可按租户/Agent 覆盖。
- 对外接口：
  - 以策略函数集合形式被 Gateway 调用：`apply_inbound(ctx, request)`, `apply_outbound(ctx, response)`。
- 配置：config/guardrail/*（toxicity.yaml、pii.yaml、quota.yaml、output.yaml）。
- 非目标：不承载编排；不访问 DB。

## 4. Buffer（Core + Plugins）
- 职责：
  - Core：定义 BufferPort/BufferPipeline/PluginBase；提供写缓冲、重试、预取、缓存框架。
  - Plugins：write/query/speculative/hybrid；RAG（chunk/encode/retrieve/rerank）作为统一插件。
- 对外接口：
  - `BufferPort.handle(request: BufferRequest) -> BufferResult`。
  - Plugin 接口：`init(config)`, `handle(req) -> result`。
- 依赖：Memory（读）、Persistence（写入异步 flush 可直接对接 Persistence）。
- 配置：config/buffer/*.yaml（插件开关/参数）。
- 非目标：不直接包含 Memory 的业务策略。

## 5. Memory（Logic + Policy）
- 职责：
  - 按层组织业务语义：
    - m0：原始交互/片段；
    - m1：语义片段/事件/事实；
    - m2：知识图谱（预留）；
    - m3：高阶汇总/程序性记忆（预留）；
    - mg：全局/群体记忆（预留）。
  - policy：时间衰减、保留/淘汰、权重策略。
- 对外接口：
  - `MemoryPort.save(...)`, `retrieve(...)`, `summarize(...)`, `decay(...)`（经 interfaces/ 暴露）。
- 依赖：Persistence（通过 StorePort）；Buffer/ Gateway 作为上游。
- 配置：config/memory/{m0,m1,m2,m3,mg,policy}.yaml。
- 非目标：不直接进行数据库操作（通过 Persistence 访问）。

## 6. Persistence（后端适配，目录名 persistence，类名保留 Store 后缀）
- 职责：
  - 统一数据访问层；实现 StorePort，适配不同后端：PgaiStore、VectorStore、GraphStore、KeywordStore 等。
- 对外接口：
  - `StorePort` 协议：`put/get/query/batch/...`；面向 Memory/Buffer 使用。
- 依赖：Database 基础设施（连接/池化）。
- 配置：config/persistence/{pgai,vector,graph,keyword}.yaml。
- 非目标：不包含业务逻辑与策略。

## 7. Database（基础设施）
- 职责：连接管理、池化、工厂、健康检查；不定义业务 Schema。
- 对外接口：`DatabaseFactory.get_connection(...)`；`health_check()`。
- 配置：config/database/{postgres,sqlite,pool}.yaml。
- 非目标：不承载 DDL 定义。

## 8. Schema（db/schema + db/schema.sql）
- 职责：唯一 Schema 来源（tables/functions/triggers/views）；db/schema.sql 为单一入口，按顺序 include 子 SQL；幂等。
- 对外接口：供工具/CI 调用：`psql -f db/schema.sql` 或脚本执行。
- 配置：无（部分 DDL 可按条件编译/变量）。
- 非目标：不提供版本迁移/回滚。

## 9. Interfaces（契约）与 Models（DTO）
- 职责：统一层间接口（Gateway/Buffer/Memory/Persistence 的 Port 与 DTO）；降低耦合。
- 对外接口：Python Protocol/ABC、Pydantic DTO。
- 配置：无。
- 非目标：不包含实现。

## 10. Monitoring（可观测性）
- 职责：统一指标/追踪/审计：QPS、P95、缓存命中、flush 延迟、召回/重排质量、拒绝率等。
- 对外接口：`monitoring.emit(metric, labels, value)`；`tracing` 钩子；审计日志 sink。
- 配置：config/server/observability.yaml（可选）。
- 非目标：不包含业务。

## 11. Config（Hydra + profiles）
- 职责：分层配置与 dev/local/prod/test 叠加；由 global_config_manager 只读提供。
- 对外接口：`config.get("gateway.pipeline")` 等；
- 结构：server/gateway/guardrail/buffer/memory/persistence/database。
- 非目标：不提供写入侧。

## 12. Scripts（工具/运维）
- 职责：launcher、数据检查、小型管理 CLI；可驱动 db/schema.sql；
- 约束：不得存放测试脚本（测试侧放在 tests/tools 或 CI）。

## 13. Tests（测试金字塔）
- 布局：unit / integration / contract / e2e / performance；fixtures/{data,config}
- 原则：新增插件/后端需通过统一契约测试；主路径以 integration 为门槛；e2e 由 launcher 驱动；性能基线定期跑。

## 14. Docs（文档）
- 职责：架构/目录/API/优化/ADR；与 refactor/* 同步迭代。
- 约定：对外/对内文档可拆分；关键设计保留 “为什么” 的取舍记录。

