# Agent 06

你是 `preprocessing_hardening_agent`。你只负责 `TASK-006`，不得实现其他任务。

任务目标：

- 清理 `preprocessing_full.py` 中的 placeholder / no-op / toy implementation

必须先读：

- `AGENTS.md`
- `preprocessing_full.py`
- `preprocessing.py`
- `analysis_service.py`
- 现有相关测试
- 上一个 agent 的 handoff

边界：

- 只做高级预处理硬化
- 高风险改动必须补测试
- 不要同时做模板自动选择，这属于 `TASK-007`

完成标准：

- 至少一条高级预处理默认路径比当前更可靠
- 明确缺依赖时的退化策略

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `template_roi_agent`。

