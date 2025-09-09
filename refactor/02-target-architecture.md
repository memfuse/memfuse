# 目标架构（Target Architecture）

主路径：API → Gateway → Buffer Layer → Memory Layer → Persistence/DB
并行护栏：Guardrail（输入/输出/策略/速率/鉴权/审计）横切贯穿 API 与 Gateway，必要时在 Buffer 入口处再次校验。

核心设计原则：
- 分层解耦：控制器（API）与业务编排（Gateway）分离，Buffer 专注“流控/缓存/聚合/预取”，Memory 专注“记忆形成/检索策略”。
- 插件化扩展：Buffer Layer 作为“横向可扩展平台”，统一抽象 RAG Layer、Code Layer 等模块为插件（Plugin）。
- 可观测与可治理：Guardrail + 监控/追踪/审计内置在 Gateway，策略配置化（Hydra）。
- 稳定的接口契约：以 Ports/Interfaces（interfaces/）定义层间契约，测试先行保证兼容性。

组件职责：
- API（FastAPI Controllers）：只做请求/响应绑定、路由、入参验证（Pydantic），业务交给 Gateway。
- Gateway（Orchestrator）：
  - 统一鉴权/鉴别身份（API Key、Tenant/Scope）、限流、请求归一化（filters/mappers）。
  - 路由到 Buffer Pipeline，注入 Guardrail 前/后置检查（毒性、PII、速率、配额）。
  - 聚合结果，统一错误模型与审计日志。
- Buffer Layer（Pipeline + Plugins）：
  - 写路径：写缓冲（聚合/分批/触发策略）、异步 Flush、重试与幂等。
  - 读路径：预取、查询缓存、混合检索（可调用 RAG 插件）、重排（可 LLM or 传统）。
  - Plugin 接口：Init(config) → handle(request: BufferRequest) → BufferResult；依赖注入模型/存储适配器。
  - 典型插件：write_buffer/query_buffer/speculative/hybrid、rag（chunk/encode/retrieve/rerank 统合）、code（将来）。
- Memory Layer：
  - M0：原始交互/片段；M1：语义/事件/事实；M2（未来）：知识图谱；策略（时间衰减、保留/淘汰）。
  - 对上提供统一 MemoryPort：save/retrieve/summarize/decay 等；对下依赖 Persistence 抽象。
- Guardrail：
  - 输入校验（schema/size/token/毒性/敏感信息）、输出约束（合规/重写/过滤）、策略化（per-tenant, per-agent）。
  - 统一挂载在 Gateway（pre/post），必要时 Buffer 入口重复关键校验。

与现有代码的映射：
- 现有 API 保留；AppService 的路由注册不变，但控制器内逻辑下沉到 Gateway。
- utils.auth 中的 RateLimitMiddleware 和鉴权拆分/增强到 gateway.auth 与 gateway.guardrails。
- `buffer/*` 保留并抽象出 `buffer/core`（接口/基类/管线）与 `buffer/plugins/*`。
- `rag/*` 归并为 `buffer/plugins/rag/*`（不改算法细节，仅改装配与入口）。
- `memory/*` 保留并扩展 policy/ 目录；与 persistence/* 依旧通过 factory + ports 解耦。
- `validators/*` 提升为 `guardrail/` 下的策略与校验器集合（API 层只做 schema 校验）。

横切关注：
- 监控与审计：统一在 `monitoring/` 暴露指标（请求速率、延迟、命中率、flush 延迟、召回率等），Gateway 负责埋点。
- 配置：统一由 `global_config_manager` 提供层间配置查询，Hydra 分环境覆盖。

