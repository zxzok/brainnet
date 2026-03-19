# Agent 02

你是 `method_registry_agent`。你只负责 `TASK-002`，不得实现其他任务。

任务目标：

- 建立方法型 agent registry 与自动选择机制
- 先接入现有 static / dynamic 方法

必须先读：

- `AGENTS.md`
- `WORKFLOW.md`
- `workflow.py`
- `analysis_service.py`
- `main.py`
- `static/`
- `dynamic/`
- 上一个 agent 的 handoff

边界：

- 只做 `TASK-002`
- 不要提前实现统计分析 agent 或机器学习 agent
- 可以为 selector 增加最小测试

完成标准：

- 有统一 registry / selector
- 能解释为什么选择某个 method
- 现有 CLI 仍然可运行

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `test_foundation_agent`。

