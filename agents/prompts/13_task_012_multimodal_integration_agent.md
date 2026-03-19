# Agent 13

你是 `multimodal_integration_agent`。你只负责 `TASK-012`，不得实现其他任务。

任务目标：

- 决定并实现 `multimodal.py` 的主链路定位

必须先读：

- `AGENTS.md`
- `README.md`
- `WORKFLOW.md`
- `multimodal.py`
- `workflow.py`
- `main.py`
- 上一个 agent 的 handoff

边界：

- 只做多模态主链路接入或重新定位
- 不要继续扩展新的统计 / ML 方法

完成标准：

- 明确 multimodal 模块是接入主 workflow，还是被降级为实验性 helper
- 文档和实现保持一致

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出最终 handoff：

- `task_id: TASK-012`
- `status`
- `changed_files`
- `validations_run`
- `residual_risks`
- `overall_program_status`

