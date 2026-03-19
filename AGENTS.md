# AGENTS.md

## 仓库目的

BrainNet 的目标是提供一个多 agent 的 MRI 影像自动化分析平台，尽可能覆盖从用户提供数据、研究背景和分析目标，到自动生成 agent workflow、自动探索分析方案、执行验证，再到结果输出的完整流程。

结合当前仓库实现和目标方向，平台主要包括：

- 用户以自然语言提供分析目标、数据来源、约束条件和成功标准
- 系统将目标拆解为可执行的 agent 工作流或任务序列
- AI agents 执行特定子任务，例如数据接入、预处理、分析、方案生成、结果比较、代码生成、执行与验证
- 系统会为不同分析方法配置专门 agent，例如统计分析 agent、机器学习 agent、常用科学数据分析 agent
- 系统根据用户目标、数据类型、约束条件和中间结果，自动选择合适的分析 agent 并组合成分析工作流
- 原始或半结构化 MRI / fMRI 数据的接入、索引与管理
- 基于本地 BIDS 数据集或 OpenNeuro 数据集的自动化分析触发
- 预处理、静态连接分析、动态连接分析、特征提取与报告生成
- 在相同目标下尝试多种分析方案，并比较哪条路径更接近目标
- 当目标已达到、搜索空间已耗尽、或证据不足时明确停手并输出结论边界
- Flask Web 界面，用于患者管理、数据浏览、任务触发与结果展示
- CLI 流水线入口，用于批量或脚本化运行分析流程

这里的“目标”可以是分析任务完成、某类模式检出、候选生物标志物发现、亚型探索或其他研究目标。但仓库和代理文档都不应把“重大科学发现”写成默认保证。更合理的表述是：平台可以支持自动生成假设、探索高价值信号、收敛出候选发现线索；最终科学结论仍需要严格验证、复现和人工审查。

本仓库当前的整理目标不是改变科学目的，也不是重写核心算法，而是让仓库更适合：

- 自治代理阅读和修改
- fresh clone 后快速安装和验证
- Symphony / Codex / harness 类工作流稳定运行

## 代码地图

### 顶层入口

- `main.py`
  主要 CLI 入口。
  负责参数解析、数据集选择、运行预处理与静态/动态分析、生成报告。
  在多 agent 场景下，它代表一个可直接调度的执行入口。

- `web_app.py`
  Flask Web 应用入口。
  负责路由、数据库初始化、任务触发、OpenNeuro 下载与分析结果展示。
  在未来或上层系统中，它也可以作为“自然语言需求提交与 workflow 触发”的交互层。

- `runtime.py`
  运行时路径与 SQLite 连接 helper。
  所有可变运行态文件都应优先从这里取路径，而不是硬编码到仓库根目录。

- `workflow.py`
  workflow 规划与任务拆解 helper。
  负责把 CLI / Web 请求转换成显式的执行步骤，便于未来多 agent 编排系统消费。

- `analysis_service.py`
  共享分析执行层。
  负责收口预处理、静态/动态分析、报告生成和特征持久化，避免 CLI 与 Web 重复实现。

### 数据与外部服务

- `data_management.py`
  BIDS 数据集索引、被试/运行发现、`DatasetManager` 封装。
  这是“从原始数据进入自动分析流程”的起点之一。

- `openneuro_client.py`
  OpenNeuro GraphQL 查询与下载逻辑。
  用于把外部公开数据集接入自动分析流程。

### 预处理与分析

- `preprocessing.py`
  较简单的预处理流水线和 ROI 提取逻辑。
  负责把原始影像转换成更适合后续分析的中间表示。

- `preprocessing_full.py`
  配置更细的模块化预处理流水线。
  是更完整的“原始影像 -> 可分析数据”处理路径。

- `static/`
  模块化静态连接分析包。

- `dynamic/`
  模块化动态连接分析包。

- `static_analysis.py`
  静态分析兼容层，保留旧导入路径。

- `dynamic_analysis.py`
  动态分析兼容层，保留旧导入路径。

### Agent 协作视角

虽然当前仓库仍然以 Python 模块和单进程入口为主，但从 agent workflow 的角度，可以把它理解为几类可协作的能力单元：

- 需求理解与任务拆解：
  当前主要由上层系统承担，未来可以对接自然语言分析请求

- 数据接入 agent：
  负责本地 BIDS 数据集发现、OpenNeuro 数据下载、数据准备

- 预处理 agent：
  负责把原始影像转换成可分析的 ROI 时序或中间结果

- 分析 agent：
  负责静态连接分析、动态连接分析、特征提取

- 统计分析 agent：
  负责传统统计检验、效应比较、显著性评估以及与目标相关的统计证据整理

- 机器学习 agent：
  负责围绕目标选择和运行合适的机器学习方法，用于模式识别、预测、分组或候选信号提取

- 常用科学数据分析 agent：
  负责更通用的数据分析步骤，例如特征探索、数据清洗、降维、聚类、相关性检查或结果汇总

- 方案探索 agent：
  负责基于用户目标组合不同分析路径、参数或方法，并控制试验顺序

- 比较与决策 agent：
  负责比较不同方案结果，判断哪个方案更接近目标，或当前证据是否不足

- 执行与验证 agent：
  负责运行 CLI、调用 Web 路径、执行测试、验证分析结果是否成功产出

- 结果整理 agent：
  负责报告生成、结果导出、可选 LLM 总结

这些角色不必一一映射为当前仓库中的独立进程，但在写文档、定义入口、增加测试或修改运行契约时，应尽量给未来的多 agent 编排留下清晰边界。

对方法型 agents 来说，重要的不只是“能运行某个方法”，还包括：

- 系统为什么选择这个 agent
- 它依赖什么输入特征或中间结果
- 它的输出如何与其他 agents 对接
- 它失败时是否会回退到其他分析方法

### 可视化与辅助组件

- `visualization.py`
  Plotly HTML 报告生成。
  是“自动分析结果输出”的主要落点之一。

- `llm_interpretation.py`
  可选 OpenAI 结果解读。

- `templates.py`
  atlas / ROI 模板加载与网络构建 helper。

- `templates/`
  Flask Jinja2 模板。

### 测试

- `tests/`
  pytest 测试目录。
  包含分析模块测试和入口 smoke tests。

## 主要入口与标准命令

优先使用这些命令，不要自己发明新的本地约定。

### 安装

基础开发环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e .[dev]
```

全量可选能力：

```bash
python3 -m pip install -e .[dev,download,hmm,llm]
```

### 标准任务

```bash
make install
make install-all
make lint
make test
make format
make run-web
make run-cli ARGS="--help"
```

### 直接入口

- Web:
  `brainnet-web`
  或 `python3 web_app.py`

- CLI:
  `brainnet-cli --help`
  或 `python3 -m brainnet.main --help`

如果上层编排系统需要在执行前做任务拆解，可以调用：

- `brainnet-cli --describe-plan /path/to/bids_dataset --subject 01 --task rest`

如果上层编排系统要驱动多个 AI agents 协作，优先把这些入口作为稳定执行边界，而不是让 agent 直接绕过入口自行拼装运行命令。

## 运行时文件与环境变量

仓库不再跟踪 SQLite 数据库。默认运行时文件位于：

- `instance/brainnet.db`
- `instance/uploads/`
- `instance/reports/`
- `instance/openneuro_datasets/`

可以通过环境变量覆盖：

```bash
export BRAINNET_INSTANCE_DIR=/tmp/brainnet-instance
export BRAINNET_DB_PATH=/tmp/brainnet-instance/brainnet.db
export BRAINNET_UPLOAD_DIR=/tmp/brainnet-instance/uploads
export BRAINNET_REPORT_DIR=/tmp/brainnet-instance/reports
export OPENNEURO_CACHE_DIR=/tmp/brainnet-instance/openneuro_datasets
```

和功能相关的其他环境变量：

- `OPENAI_API_KEY`

如果代理需要在测试中隔离运行态，请优先设置 `BRAINNET_INSTANCE_DIR`，而不是修改仓库内路径。

## 可安全编辑的区域

通常可以直接编辑：

- 顶层 Python 模块
- `dynamic/`
- `static/`
- `tests/`
- `templates/`
- 文档文件，例如 `README.md`、`AGENTS.md`、`WORKFLOW.md`
- CI 与开发配置文件，例如 `pyproject.toml`、`Makefile`、`.github/workflows/*`

## 不应提交的生成产物

以下内容应视为运行产物或本地缓存，不应提交：

- `instance/`
- `uploads/`
- `reports/`
- `openneuro_datasets/`
- `__pycache__/`
- `.pytest_cache/`
- `.ruff_cache/`
- `.coverage`
- `htmlcov/`
- `.venv/`
- `venv/`
- 操作系统元数据文件，例如 `.DS_Store`

## 高风险改动区域

以下改动需要更高谨慎度，并且通常需要补测试或至少手动验证：

### 1. 数据库 schema 与持久化

- `web_app.py` 中的 `init_db()`
- 任何 SQL 表结构
- 特征写入逻辑

风险：

- 旧数据库兼容性
- Web 页面查询失败
- 结果展示字段错位

### 2. 预处理和分析逻辑

- `preprocessing.py`
- `preprocessing_full.py`
- `static/`
- `dynamic/`

风险：

- 数值结果变化
- 输出 shape 变化
- 测试 fixture 失效
- 报告层和数据库层不再兼容

### 3. 入口与运行契约

- `main.py`
- `web_app.py`
- `runtime.py`
- `pyproject.toml`
- `Makefile`

风险：

- CLI / Web 无法启动
- CI 失败
- 代理无法在 fresh clone 中完成 install/run/test

### 4. OpenNeuro 与外部依赖

- `openneuro_client.py`
- `data_management.py`

风险：

- 网络调用不可复现
- 下载路径污染工作区
- 测试误用真实网络

## 代理工作原则

### 总体原则

- 保持科学目的不变
- 保持“用户提供数据和目标 -> agent workflow -> 多方案探索 -> 自动执行 -> 验证 -> 结果输出”的平台方向不变
- 保持“专门分析 agent 按方法分工，并由系统自动选择生成 workflow”的平台方向不变
- 优先做最小安全改动
- 不要无故重命名核心模块
- 不要把运行态文件重新提交回仓库
- 不要伪造缺失依赖下的功能
- 不要把候选线索、弱证据或单次运行结果表述成已确认的重大科学发现

### 发现问题时的优先级

优先修复：

- 无法安装
- 无法导入
- 无法启动 CLI / Web
- 测试不稳定或不可重复
- 路径硬编码导致运行态污染仓库
- 打断多 agent 工作流中的某个关键执行阶段

其次再处理：

- 文档不一致
- 配置缺失
- 轻量结构整理

## 改动前应该先看什么

如果任务是：

- Web 路由/页面问题：
  看 `web_app.py`、相关 `templates/*.html`、`tests/test_entrypoints.py`

- CLI 问题：
  看 `main.py`、相关分析模块、`tests/test_entrypoints.py`

- 数据集索引 / OpenNeuro：
  看 `data_management.py`、`openneuro_client.py`、对应测试

- 预处理/分析结果：
  看 `preprocessing.py`、`preprocessing_full.py`、`static/`、`dynamic/` 和对应测试

- 运行态 / SQLite / 上传目录：
  看 `runtime.py`、`web_app.py`

- 自然语言需求编排、agent workflow 承接、自动执行链路：
  先看入口契约文件、`README.md`、`AGENTS.md`、`WORKFLOW.md`，再看 `main.py` / `web_app.py`

- 目标驱动探索、方案比较、自动停手条件：
  先看 `README.md`、`AGENTS.md`、`WORKFLOW.md`，再看 `main.py`、`web_app.py`、分析模块和测试

- 统计分析 agent、机器学习 agent、科学数据分析 agent 的选择与衔接：
  先看 `README.md`、`AGENTS.md`、`WORKFLOW.md`，再看分析模块、入口文件和相关测试

## 验证要求

### 最低要求

任何非纯文档改动之后，至少跑：

```bash
make lint
make test
```

### 按改动类型追加验证

- 改了 `main.py`：
  额外确认 `brainnet-cli --help` 或 `python3 -m brainnet.main --help` 可用

- 改了 `web_app.py` 或运行时路径：
  额外确认 Flask smoke test 通过

- 改了数据库写入或 schema：
  至少确认测试环境会创建新的 SQLite 文件，并且首页或相关 API 不报错

- 改了分析模块：
  跑相关目标测试，必要时再跑全量

- 改了可选依赖功能：
  不要让基础安装路径变重；基础 `.[dev]` 仍应可用

- 改了 workflow、任务编排或 agent 契约文档：
  同步检查 `README.md`、`AGENTS.md`、`WORKFLOW.md`、`CLAUDE.md` 之间描述是否一致
  并确认文档里对“目标达成”“证据不足”“人工复核”的边界描述没有互相冲突
  也要确认“分析方法 -> 专门 agent -> workflow 选择逻辑”的描述没有漂移

## 测试与 fixture 约定

- 测试应尽量使用 `tmp_path`
- 测试中的数据库应写到临时 `instance/`
- 测试不得依赖仓库内预置 SQLite 数据
- OpenNeuro 相关测试优先 mock，不要打真实网络
- 如果第三方包是可选依赖，使用 `pytest.importorskip(...)`

## 文档更新规则

如果你修改了以下内容，通常也应该更新文档：

- 安装方式
- CLI 参数
- Web 运行方式
- 新增环境变量
- 新增标准验证步骤
- 运行态目录或产物位置

至少检查：

- `README.md`
- `AGENTS.md`
- `WORKFLOW.md`

## 代理输出期望

完成任务后，输出应至少包含：

- 改了什么
- 为什么改
- 跑了哪些验证
- 有哪些残余风险或未处理项

如果任务属于多 agent 平台能力建设，额外说明：

- 这次改动影响了 workflow 的哪个阶段
- 是增强了需求理解、任务编排、方法选择、方案探索、自动执行、验证，还是结果分析
- 对未来 agent 协作边界是否有新约束
- 当前是让系统更容易达到目标，还是更容易识别“证据不足应停手”

如果信心不足，必须明确说明：

- 哪个假设不确定
- 哪个依赖缺失
- 哪一步无法验证
- 建议人工接手关注什么
