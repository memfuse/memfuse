# 测试与质量策略（Testing Strategy）

测试金字塔：
- Unit：对 Ports/Adapters/Plugins 的行为进行细粒度测试（mock 下层依赖）。
- Integration：跨层主路径（API→Gateway→Buffer→Memory→Persistence）的组合验证；提供最小可用后端（sqlite/qdrant dummy）。
- Contract：REST 资源合同（路径、参数、状态码、响应模式），与 OpenAPI 校验。
- E2E：`scripts/memfuse_launcher.py` 启动 → 健康检查 → 核心流程（写入、检索）。
- Performance：关键 KPI（吞吐、P95 延迟、缓存命中、flush 延迟、召回率/重排质量）。

具体建议：

脚本与数据组织：
- 测试相关脚本不放在 `scripts/` 目录；统一置于 `tests/tools` 或 CI 配置中。
- 小规模测试数据与配置集中在 `tests/fixtures/{data,config}`，并附来源与 LICENSE 说明。

- 为 Gateway/Guardrail/BufferPipeline 定义可注入的端到端 Test Harness（可在 tests/integration/* 复用）。
- 为 Plugin 设计统一的契约测试（新增插件必须通过同一组测试）。
- 在 CI 中：分层并行执行（unit→integration→contract→e2e），失败快速反馈；性能在定时任务跑。
- 覆盖率阈值先适中（例如 80%），逐阶段提升。

