# BrainNet Sequential Agents

本目录提供一套面向 BrainNet 的顺序执行 agent 任务包。

设计原则：

- 一个 agent 只完成一个任务
- agent 必须按顺序执行
- 当前 agent 完成前，不得开始下一个 agent
- 当前 agent 只能处理自己负责的任务范围，不能顺手实现后续任务
- 每个 agent 完成后必须产出 handoff summary，供下一个 agent 使用

## 文件结构

- `sequential_agents.yaml`
  顺序执行清单，定义 agent 顺序、任务映射、输入输出和 handoff 要求。

- `orchestration.md`
  顶层执行规则，说明如何串行运行这些 agents。

- `prompts/*.md`
  每个 agent 的单任务 prompt。一个文件对应一个 agent。

## 执行规则

1. 先阅读：
   - `AGENTS.md`
   - `WORKFLOW.md`
   - `docs/IMPLEMENTATION_GAP_AUDIT.md`
   - `docs/DEVELOPMENT_TASK_PROMPTS.md`

2. 再按 `sequential_agents.yaml` 中的 `order` 串行执行。

3. 每个 agent 必须：
   - 只处理自己的 `task_id`
   - 只实现最小安全改动
   - 输出本任务完成状态
   - 输出 handoff summary
   - 不得提前修改下一任务的目标区域，除非是当前任务完成所必需的最小配套改动

4. 如果某个 agent 被阻塞：
   - 必须明确记录阻塞原因
   - 不得跳过实现细节直接进入下一个 agent
   - 只能在 handoff 中建议后续人工或新 agent 处理

## 推荐运行方式

建议一次只调用一个 agent prompt，例如：

- `agents/prompts/01_task_001_workflow_runtime_agent.md`
- `agents/prompts/02_task_002_method_registry_agent.md`
- `agents/prompts/03_task_013_test_foundation_agent.md`

推荐优先顺序已经编码进 `sequential_agents.yaml`。

