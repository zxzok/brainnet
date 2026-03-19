---
tracker:
  issue_id: "<issue-id>"
  title: "<issue-title>"
  reference: "<issue-url-or-ticket>"
  labels: []
workspace:
  root: "<repo-root>"
  branch: "<working-branch>"
  python: "python3"
  venv: ".venv"
  notes: "<local-workspace-notes>"
hooks:
  before_read: []
  before_edit: []
  after_edit: []
  before_validate: []
  before_handoff: []
agent:
  role: "senior-coding-agent"
  change_strategy: "smallest-safe-change"
  preserve_scientific_behavior: true
  stop_when_uncertain: true
  confidence_threshold: "<set-by-runner>"
codex:
  install:
    - "python3 -m venv .venv"
    - "source .venv/bin/activate"
    - "python3 -m pip install --upgrade pip"
    - "make install"
  validate:
    - "make lint"
    - "make test"
  summary_template:
    changed_files: []
    validations_run: []
    risks: []
    follow_ups: []
---

# BrainNet Workflow

本文件定义 BrainNet 在 Symphony / Codex / harness 场景下的推荐执行流程。

BrainNet 的仓库目标是提供一个多 agent 的 MRI 影像自动化分析平台，尽可能覆盖从用户提供数据、研究背景和分析目标，到 agent workflow 自动生成、自动探索分析方案、执行验证，再到结果输出的完整流程。

当前仓库已经提供了一个轻量的代码级 workflow 规划层：

- `brainnet.workflow.AnalysisRequest`
- `brainnet.workflow.build_analysis_plan(...)`
- `brainnet-cli --describe-plan`

它的职责不是替代未来的多 agent orchestrator，而是把入口请求转换成稳定、可检查的步骤定义，避免任务拆解只停留在文档里。

因此，代理在执行任务时应始终优先保护这条主流程：

用户提供数据和目标 -> 任务拆解 / workflow 生成 -> 候选分析方案生成 -> 数据接入 -> 预处理 -> 静态 / 动态分析执行 -> 结果比较 / 目标评估 -> 特征写入 -> 验证 -> 报告 / Web 展示 / 结果分析

目标不是最大化重构，而是在保持科学目的和现有模块边界基本稳定的前提下，做最小、安全、可验证的改动。

## 0. 多 Agent 平台视角

在 BrainNet 的目标架构里，系统应支持多个 AI agents 协作完成不同阶段的任务。典型职责可以理解为：

- 需求理解 agent：
  把用户自然语言分析请求转成结构化任务

- workflow 编排 agent：
  生成任务顺序、选择入口、决定需要调用哪些模块或命令

- 方案探索 agent：
  基于目标生成多个候选分析方案，控制试验次序、参数范围和回退路径

- 统计分析 agent：
  承担统计检验、效应评估、显著性分析或与科学假设直接相关的统计分析方法

- 机器学习 agent：
  承担分类、回归、聚类、表征学习或其他为目标服务的机器学习分析方法

- 常用科学数据分析 agent：
  承担特征探索、降维、相关性分析、数据质量检查和常见科学数据处理任务

- 编码 agent：
  在平台能力不足时补充代码、模板、配置或 glue logic

- 执行 agent：
  调用 CLI、Web 路径、数据下载、分析流水线

- 比较与决策 agent：
  比较不同分析方案的结果，判断是否达到目标，或当前证据是否不足

- 验证 agent：
  跑测试、检查日志、确认结果文件和数据库记录

- 结果分析 agent：
  汇总报告、解释结果、生成面向用户的输出

即使当前仓库尚未完整实现这一层编排，文档、入口和改动方式都应尽量服务于这种多 agent 协作模式。

在目标架构里，系统不应把所有分析都压到一个“万能分析 agent”上，而应按分析方法维护清晰分工，再根据用户目标、数据约束和已有中间结果，自动选择哪些方法型 agents 进入 workflow。

这里的“达到目标”不应被狭义理解为任意一次运行产出结果文件，而应理解为：当前探索路径已经得到足够强的、与用户目标一致的结果信号，或者至少已经明确哪些方案失败、哪些方案值得继续。对于“重大科学发现”这类高风险表述，代理只能把输出描述为候选发现、假设线索或需要进一步验证的结果，不能直接写成已确认结论。

## 1. 先读任务，再缩小范围

收到任务后先做三件事：

1. 读清楚任务到底是在修：
   - 自然语言需求承接或 workflow 编排
   - 方法型 agent 的选择、映射或组合
   - Web 交互
   - CLI 行为
   - 数据索引或 OpenNeuro
   - 预处理 / 静态分析 / 动态分析
   - 自动编码、执行、验证或结果分析
   - 运行时路径、数据库、文档或 CI
2. 明确哪些模块最可能受影响。
3. 只打开必要文件，不要一开始就做全仓大改。

优先定位入口：

- Web 问题从 `web_app.py` 开始
- CLI 问题从 `main.py` 开始
- 数据访问问题从 `data_management.py` / `openneuro_client.py` 开始
- 分析问题从 `preprocessing*.py`、`static/`、`dynamic/` 开始
- 路径/数据库问题从 `runtime.py` 和 `web_app.py` 开始

## 2. 先理解当前行为，再编辑

编辑前至少确认以下几点中的相关项：

- 当前的标准安装命令是什么
- 当前的标准验证命令是什么
- 相关测试是否已经存在
- 该功能是否依赖可选依赖
- 该改动是否会影响运行态文件位置
- 该改动是否会影响数据库 schema 或特征写入

如果任务涉及已有 bug，不要只看 README，优先看真实代码路径和现有测试。

如果任务会影响“自然语言需求 -> 自动 workflow -> 自动执行 -> 结果输出”的主流程，必须先明确它会落在哪个阶段，再决定修改范围。

如果任务和“给定目标后自动尝试多种分析方案”有关，还要进一步明确：

- 目标是分析成功率、结果质量、统计证据，还是报告可解释性
- 候选方案分别依赖哪些统计分析 agent、机器学习 agent 或通用科学数据分析 agent
- 系统如何比较多个候选方案
- 什么时候应该继续探索，什么时候应该明确停手

如果需要在执行前把当前请求拆成显式步骤，优先复用 `build_analysis_plan(...)` 或 CLI 的 `--describe-plan`，不要在外部系统里重新硬编码一套不同的阶段命名。

## 3. 做最小安全改动

改动时遵循这些原则：

- 优先做局部修复，不做无关重构
- 不改变科学目标或算法意图，除非任务明确要求
- 不把运行态产物重新放回版本库
- 不在缺少依赖时伪造分析结果
- 不为了“好看”而改动大量文件
- 不要无意打断多 agent 自动分析流程中的任一关键阶段
- 不要把“候选信号”直接升级成“重大科学发现”式断言
- 不要把方法选择逻辑隐藏在难以观察的隐式分支里

如果需要新增文件，优先新增：

- 测试
- 文档
- 小型 helper
- 配置文件

而不是拆大模块，除非任务本身就是结构治理。

## 4. 编辑时的仓库约束

### 运行态约束

以下内容必须保持为本地运行态，而不是提交产物：

- `instance/`
- `uploads/`
- `reports/`
- `openneuro_datasets/`

默认情况下，这些运行态目录现在都由 `runtime.py` 统一解析；报告目录默认为 `instance/reports/`，OpenNeuro 缓存目录默认为 `instance/openneuro_datasets/`，并可通过环境变量覆盖。

如果任务涉及数据库、上传或报告输出，优先使用：

- `BRAINNET_INSTANCE_DIR`
- `BRAINNET_DB_PATH`
- `BRAINNET_UPLOAD_DIR`

### 可选依赖约束

以下能力是可选的，不应阻塞基础开发路径：

- `hmmlearn`
- `openai`
- `datalad`
- FSL / ANTs / SPM

基础开发安装应仍然可以用 `make install` 完成。

## 5. 选择验证策略

默认验证：

```bash
make lint
make test
```

在此基础上按改动类型附加验证：

### CLI 相关改动

至少再确认：

```bash
brainnet-cli --help
```

或：

```bash
python3 -m brainnet.main --help
```

### Web 相关改动

至少确认：

- Flask 应用可导入
- 相关 smoke test 通过
- 首页或对应路由不会立刻 500

### 数据库 / schema 相关改动

至少确认：

- 临时环境会创建新的 SQLite 文件
- 旧逻辑不会因为缺少表而立即崩溃
- 如果改了表结构，相关读取/写入路径都同步更新

### 分析相关改动

至少确认：

- 对应模块测试通过
- 输出 shape / 类型 /字段名没有意外破坏
- 可选依赖缺失时行为仍然明确
- 如果改动影响主流程，确认不会阻断“自然语言需求 -> workflow -> 自动分析结果”链路中的下游步骤

### Workflow / Agent 平台相关改动

至少确认：

- 文档对多 agent 角色分工的描述仍一致
- 入口命令仍然适合被上层编排系统调用
- 自动编码、执行、验证、结果分析这些阶段之间没有新增隐式耦合
- 仓库没有把 agent 协议写死成只能服务某个外部平台
- 目标输入、方案探索、结果比较和停手条件的描述是明确的
- 统计分析 agent、机器学习 agent、科学数据分析 agent 的职责边界仍清晰
- 方法到 agent 的选择逻辑或映射关系仍然可解释、可验证
- 如果文档提到科学发现，只能表述为候选线索、候选发现或需人工复核的结果

## 6. 什么时候应该停手并交接

出现以下情况时，不要硬猜：

- 任务要求改变科学结论或分析定义，但仓库内没有足够依据
- 缺少关键外部依赖且无法验证
- 需要真实数据集或外部工具才能确认结果正确
- 多种修法都可能成立，但没有清晰业务偏好
- 继续修改会引发大范围结构重写
- 结果看起来“很有意思”，但没有足够证据支持强结论

此时应该输出：

- 已经确认的事实
- 当前阻塞点
- 你不确定的假设
- 建议人工决策的选项
- 当前结果更像候选发现、弱证据，还是可复现的稳定结论

## 7. 完成后的交付格式

交付时应包含四部分：

### 已完成工作

- 改了哪些文件
- 每个改动的目的是什么

### 验证

- 跑了哪些命令
- 哪些通过，哪些跳过

### 风险

- 仍未覆盖的路径
- 依赖外部环境的部分

### 后续建议

- 值得继续做但本次刻意没有展开的改进项

## 8. 推荐执行顺序

可按下面顺序执行一次完整任务：

1. 阅读 issue / ticket / task 描述
2. 打开相关入口文件和对应测试
3. 明确最小改动面
4. 实施改动
5. 先跑相关小范围验证
6. 再跑 `make lint` 和 `make test`
7. 汇总结果、风险和后续事项

## 9. 对 BrainNet 特别重要的注意点

- `web_app.py` 既有路由又有数据库逻辑，改动时要特别小心副作用
- `main.py` 是自治代理最常使用的入口之一，参数和导入路径必须保持稳定
- `runtime.py` 是运行态隔离的关键，不要绕过它重新硬编码数据库路径
- `static_analysis.py` 和 `dynamic_analysis.py` 是兼容层，改动时注意不要无意破坏旧导入路径
- OpenNeuro 相关逻辑优先 mock 测试，不要把真实网络调用混入单元测试
- 如果新增“目标评估”或“方案比较”逻辑，务必让停手条件和失败条件显式可见
- 对任何涉及端到端流程的改动，都要优先考虑是否仍然支持“自然语言分析需求驱动多 agent 自动完成分析”的仓库目标
