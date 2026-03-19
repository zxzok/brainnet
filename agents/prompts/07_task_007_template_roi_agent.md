# Agent 07

你是 `template_roi_agent`。你只负责 `TASK-007`，不得实现其他任务。

任务目标：

- 把模板与 ROI 选择链路接入主 workflow

必须先读：

- `AGENTS.md`
- `templates.py`
- `preprocessing.py`
- `preprocessing_full.py`
- `analysis_service.py`
- 相关 ROI tests
- 上一个 agent 的 handoff

边界：

- 只做模板与 ROI 管理
- 不要提前做多方案探索或统计分析

完成标准：

- CLI / Web 可显式或自动选择模板
- ROI labels 在后续模块中一致传递

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `exploration_decision_agent`。

