# Agent 01

你是 `workflow_runtime_agent`。你只负责 `TASK-001`，不得实现其他任务。

任务来源：

- `docs/IMPLEMENTATION_GAP_AUDIT.md`
- `docs/DEVELOPMENT_TASK_PROMPTS.md`

任务目标：

- 实现目标输入与工作流执行模型
- 把 workflow 从静态 plan 扩展为运行时可跟踪状态

必须先读：

- `AGENTS.md`
- `WORKFLOW.md`
- `workflow.py`
- `main.py`
- `web_app.py`
- `runtime.py`
- `tests/test_runtime_and_workflow.py`

边界：

- 只做 `TASK-001`
- 不要顺手实现 `TASK-002` 或 `TASK-003`
- 可以补当前任务必需的测试和最小文档

完成标准：

- 明确 workflow run / step run / status / timestamps / error
- CLI 或 Web 至少一处可查询 workflow run 状态
- 不破坏现有入口

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`
- `source .venv/bin/activate && python3 -m brainnet.main --help`

结束时必须输出 handoff：

- `task_id: TASK-001`
- `status`
- `changed_files`
- `validations_run`
- `residual_risks`
- `next_agent_notes` 给 `method_registry_agent`

