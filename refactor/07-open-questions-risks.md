# 风险与开放问题（Risks & Open Questions）

主要风险：
- Gateway/Validators 目录目前无源码：
  - 风险：历史实现分散在 utils/services 中，迁移时需梳理职责与引用，避免回归。
  - 缓解：先引入最小 orchestrator，逐步搬运逻辑；以集成/合同测试兜底。
- 配置双轨（legacy vs global）：
  - 风险：行为不一致、变更难以追踪。
  - 缓解：统一到 global，保留只读 proxy 适配 1-2 个版本，逐步移除。
- RAG 与 Buffer 的耦合：
  - 风险：迁移到 Plugin 接口涉及大量 import 路径变化。
  - 缓解：在 buffer/core 提供适配层，先支持旧入口，最终切换到插件接口。
- 性能回归：
  - 风险：Gateway 新增护栏/变换可能提高 P95 延迟。
  - 缓解：开启按阶段的性能基线对比与指标告警；可开关策略。

开放问题：
- Guardrail 策略的边界：哪些策略需要在 Gateway 强制？哪些在 Buffer/Memory 再次校验？
- Memory 的时间衰减与事实融合策略细节：是否按 tenant/agent 细化？
- 插件市场化：是否需要标准化插件描述（manifest）与动态加载机制？
- OpenAPI 与合同测试的生成/校验自动化程度：是否纳入 CI 必选？

