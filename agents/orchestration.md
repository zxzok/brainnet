# Sequential Agent Orchestration

## 目标

本文件定义如何按顺序运行 BrainNet 的单任务 agents。

这些 agents 的来源是：

- `docs/IMPLEMENTATION_GAP_AUDIT.md`
- `docs/DEVELOPMENT_TASK_PROMPTS.md`

## 核心约束

- 严格串行执行
- 一次只运行一个 agent
- 一个 agent 只完成一个任务
- 当前 agent 完成并产出 handoff 后，才能启动下一个 agent

## Handoff 格式

每个 agent 在结束时都应输出以下内容：

1. `task_id`
2. `status`
   - `completed`
   - `partial`
   - `blocked`
3. `changed_files`
4. `validations_run`
5. `residual_risks`
6. `next_agent_notes`

## 串行顺序

推荐顺序：

1. `TASK-001` 目标输入与工作流执行模型
2. `TASK-002` 方法型 agent registry 与自动选择
3. `TASK-013` 端到端测试补齐
4. `TASK-008` Web 分析页和特征页修复
5. `TASK-009` MRI 与网络可视化接真实数据
6. `TASK-006` 高级预处理替换 placeholder
7. `TASK-007` 模板与 ROI 选择链路
8. `TASK-003` 多方案探索、比较与停手机制
9. `TASK-004` 统计分析 agent
10. `TASK-005` 机器学习 agent
11. `TASK-010` 报告与 LLM 解读硬化
12. `TASK-011` OpenNeuro 任务可靠性
13. `TASK-012` 多模态主链路接入

## 不允许的行为

- 在 `TASK-001` 中顺手实现 `TASK-002`
- 在修 UI 页面时顺手引入新的统计分析模块
- 因为“看起来相关”而跳过顺序
- 在 handoff 不完整的情况下启动下一 agent

## 允许的最小配套改动

以下情况允许小范围跨文件改动：

- 当前任务必须补一条测试才能验证
- 当前任务必须补一条文档说明才能保证运行契约一致
- 当前任务必须做最小接口适配，才能让已有入口继续工作

但即使如此，也不能把下一个任务的主体实现提前做掉。

