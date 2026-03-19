# Agent 10

你是 `machine_learning_agent`。你只负责 `TASK-005`，不得实现其他任务。

任务目标：

- 为 BrainNet 增加机器学习 agent

必须先读：

- `AGENTS.md`
- `WORKFLOW.md`
- `analysis_service.py`
- `static/`
- `dynamic/`
- 上一个 agent 的 handoff

边界：

- 只做机器学习 agent
- 不要再扩统计分析 agent

完成标准：

- 有 feature matrix builder 和 baseline ML 流程
- 有验证指标和结构化输出
- 接入现有 workflow / registry

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `reporting_llm_agent`。

