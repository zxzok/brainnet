# Agent 05

你是 `visualization_data_agent`。你只负责 `TASK-009`，不得实现其他任务。

任务目标：

- 替换 MRI 与网络可视化的示例数据和 dummy API

必须先读：

- `AGENTS.md`
- `web_app.py`
- `templates/mri_visualization.html`
- `templates/network_visualization.html`
- `templates/features_detail.html`
- 上一个 agent 的 handoff

边界：

- 只做 `TASK-009`
- 如果 MRI 页面暂时无法接真实数据，可以明确降级或下线，但不要继续保留 `example.com`

完成标准：

- `/api/network_data` 不再返回硬编码 dummy graph
- `/mri` 不再写死示例 URL

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `preprocessing_hardening_agent`。

