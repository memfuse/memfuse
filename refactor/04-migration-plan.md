# 迁移计划（Phased Migration Plan）

总体策略：并行目录+适配层，避免一次性大改；每阶段以测试套件与脚本启动作为质量门槛。

Phase 0 — 准备与对齐（本阶段输出即本目录）
- 确认目标架构与目录方案，冻结“接口契约”草案（interfaces/）。
- 统一配置入口：以 `global_config_manager` 为准，保留 legacy 只读过渡。
- 风险评估与里程碑定义。

Phase 1 — Gateway 抽取与 Guardrail 落地
- 从 `api/*` 与 `utils.auth` 抽取/迁移到 `gateway/`：
  - request_router、transformers、auth（API Key/tenant/scope）、guardrails（输入/输出策略）。
  - AppService 仅保留路由注册，controller 内业务调用 Gateway。
- 引入基础审计/指标埋点（请求耗时、QPS、限流命中、拒绝率）。
- 通过 unit/integration 测试验证 API 行为不变；契约测试必须通过。

Phase 2 — Buffer 插件化与 RAG 归一
- 定义 `buffer/core` 接口（PluginBase/BufferPort/BufferPipeline）。
- 将 `buffer/*` 既有实现迁至 `buffer/plugins/*`，按写/读/混合划分，flush 与队列策略抽象到 core。
- 将 `rag/*` 迁至 `buffer/plugins/rag/`，用统一 Plugin 接口接入（算法不改，入口与装配改）。
- 完成缓存/预取/重排策略在 Pipeline 内的统一编排，暴露性能指标。

Phase 3 — Memory 层策略与 Store 解耦加固
- `memory/*` 增加 policy/（时间衰减/保留/权重策略）。
- `interfaces/` 明确 MemoryPort；`store/*` 只通过 factory + port 暴露。
- 扩展/标准化多后端（pgai/vector/graph/keyword）的健康检查与 fallback。

Phase 4 — 清理与文档/测试收尾
- 移除旧路径与适配层；统一 imports。
- 完整更新 docs/architecture 与 api 文档。
- 性能回归基准（tests/performance/*）与 e2e 验证通过。

质量门槛（每阶段均需满足）：
- 所有 unit/integration/contract/smoke 通过；关键 e2e 场景通过。
- `poetry run python scripts/memfuse_launcher.py` 正常启动、健康检查通过。
- OpenAPI 与合同测试一致。

