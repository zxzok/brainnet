# Agent 04

你是 `web_analysis_features_agent`。你只负责 `TASK-008`，不得实现其他任务。

任务目标：

- 修复 `/analysis`
- 修复 `/features`
- 清理明显占位内容

必须先读：

- `AGENTS.md`
- `web_app.py`
- `templates/analysis.html`
- `templates/features.html`
- `templates/features_detail.html`
- `tests/`
- 上一个 agent 的 handoff

边界：

- 只做分析页和特征页
- 不要把网络可视化和 MRI 可视化一起做掉，那属于 `TASK-009`

完成标准：

- `/analysis` 不再展示示例图
- `/features` 不再依赖缺失上下文
- 增加对应 route tests

验证：

- `source .venv/bin/activate && make lint`
- `source .venv/bin/activate && make test`

结束时必须输出 handoff 给 `visualization_data_agent`。

