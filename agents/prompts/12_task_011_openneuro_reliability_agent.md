# Agent 12

你是 `openneuro_reliability_agent`。你只负责 `TASK-011`，不得实现其他任务。

任务目标：

- 提升 OpenNeuro 下载和分析任务链路的可靠性

必须先读：

- `AGENTS.md`
- `web_app.py`
- `openneuro_client.py`
- `data_management.py`
- `runtime.py`
- 上一个 agent 的 handoff

边界：

- 只做 OpenNeuro 任务可靠性
- 去掉隐式 fallback
- 测试优先 mock，不打真实网络

完成标准：

- 有更明确的任务状态
- legacy fallback 被清理或显式化

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `multimodal_integration_agent`。

