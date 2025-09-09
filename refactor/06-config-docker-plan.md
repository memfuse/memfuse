# 配置与容器/编排方案（Config & Docker Plan）

配置统一化：
- 统一入口：`global_config_manager`；API/Gateway/Buffer/Memory/Persistence 通过 ports 读取。
- Hydra 分层与 Profile：`dev/local/prod/test` 叠加，按层拆分（gateway/buffer/memory/persistence/server）。
- 环境变量约定：前缀 `MEMFUSE_`，launcher 兼容读取。

Docker/Compose：
- 保持 `docker/compose/*.yml` 多环境，命名与健康检查标准化。
- App 容器环境变量直连 Hydra 配置（只保留必要变量）。
- 可选后端组件以覆盖文件形式提供（qdrant、neo4j）；默认仅 TimescaleDB。
- 优化脚本集中在 `docker/scripts/*`（保持 launcher 的外部调用接口稳定）。

Scripts 目录定位：
- `scripts/` 仅用于通用工具/运维集成（例如启动器、数据检查、小型管理 CLI）。
- 测试相关脚本不得放在 `scripts/` 下；一律置于 `tests/tools` 或 CI 配置中。


Launcher 兼容性：
- 不改变 `poetry run python scripts/memfuse_launcher.py` 使用方式与输出体验。
- 仅在内部：兼容新 Gateway/Buffer 初始化顺序；健康检查 endpoint 保持 `/api/v1/health`。

