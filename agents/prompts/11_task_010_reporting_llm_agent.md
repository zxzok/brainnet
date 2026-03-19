# Agent 11

你是 `reporting_llm_agent`。你只负责 `TASK-010`，不得实现其他任务。

任务目标：

- 硬化报告与 LLM 解读链路

必须先读：

- `AGENTS.md`
- `README.md`
- `visualization.py`
- `llm_interpretation.py`
- 上一个 agent 的 handoff

边界：

- 只做报告与解读
- 不要顺手实现 OpenNeuro 可靠性任务

完成标准：

- 输出口径改为候选线索 / 需人工复核
- 更新过时模型调用
- 明确 provenance 或元数据

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `openneuro_reliability_agent`。

