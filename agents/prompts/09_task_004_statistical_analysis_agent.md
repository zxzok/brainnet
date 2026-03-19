# Agent 09

你是 `statistical_analysis_agent`。你只负责 `TASK-004`，不得实现其他任务。

任务目标：

- 为 BrainNet 增加传统统计分析 agent

必须先读：

- `AGENTS.md`
- `WORKFLOW.md`
- `analysis_service.py`
- `static/`
- `dynamic/`
- 上一个 agent 的 handoff

边界：

- 只做统计分析 agent
- 不要同时实现机器学习 agent

完成标准：

- 有独立模块、配置和测试
- 可被 workflow 选择并产出结构化结果

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `machine_learning_agent`。

