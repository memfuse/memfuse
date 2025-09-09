# MemFuse 大重构（Refactor）工作区

本目录用于本次架构重构的所有规划文档与产出物（仅文档，不涉及代码实施）。

目标：在不立即深入 Coding 的前提下，完成系统化的架构设计与目录重构规划，形成可执行、可落地的迁移路线图。

文档索引：
- 01-current-state.md：现状评估与问题清单
- 02-target-architecture.md：目标架构（API / Gateway / Buffer Layer / Memory Layer / Guardrail）
- 03-directory-structure-plan.md：目录与模块重构方案（src / config / docker / docs / tests 等）
- 04-migration-plan.md：分阶段迁移计划与里程碑
- 05-testing-strategy.md：测试分层与质量门禁策略
- 06-config-docker-plan.md：配置统一化与容器/编排方案
- 07-open-questions-risks.md：风险清单与开放问题

注意：
- 文档以“实际代码”为准，现有部分文档可能滞后或不准确。
- 重构期间坚持“最小可行迁移（MVP）+ 渐进替换”的策略，避免大爆炸式重写。

