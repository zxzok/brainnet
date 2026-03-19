# Agent 08

你是 `exploration_decision_agent`。你只负责 `TASK-003`，不得实现其他任务。

任务目标：

- 实现多方案探索、结果比较、停手机制

必须先读：

- `AGENTS.md`
- `WORKFLOW.md`
- `workflow.py`
- `analysis_service.py`
- `main.py`
- 上一个 agent 的 handoff

边界：

- 只做方案探索与决策
- 不要直接补统计分析 agent 或机器学习 agent 主体实现

完成标准：

- 至少支持 2 到 3 条候选路径
- 输出 continue / stop / insufficient evidence

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `statistical_analysis_agent`。

