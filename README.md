# BrainNet — 功能磁共振脑网络动力学分析计算平台

BrainNet 是一个面向科研人员的 fMRI 脑网络分析平台，提供从数据获取、预处理、静态/动态功能连接分析到交互式可视化的完整工作流。平台内置深色科学主题 Web 界面，支持被试管理、OpenNeuro 公共数据集下载、自动化分析流水线和 Claude AI 智能对话。

## 功能特性

### 数据管理
- BIDS 格式数据集自动索引与浏览
- OpenNeuro 公共数据集搜索、下载与内容浏览
- 被试管理（增删改查）与 MRI 影像上传

### 预处理流水线
- 空间平滑、时间带通滤波、运动校正
- 去噪回归（OLS 混杂信号去除）
- ROI 时间序列提取（支持 AAL / Harvard-Oxford / Schaefer 图谱）

### 静态连接分析
- Pearson 相关功能连接矩阵
- 图论指标：度、聚类系数、全局效率、模块度、介数中心性等

### 动态连接分析
- 滑动窗口 + K-means 聚类状态识别
- 高斯隐马尔可夫模型（HMM）
- 共激活模式分析（CAP）
- 动态指标：状态占用率、驻留时间、转移概率矩阵

### 可视化与报告
- 连接矩阵热图、动态状态时间线
- 状态占用率/驻留时间柱状图
- 交互式 HTML 分析报告（Plotly）

### AI 智能对话
- 集成 Claude (Anthropic) 大语言模型
- 通过 Web 界面配置 API 密钥，无需环境变量
- 自然语言分析结果解读

### Web 界面
- 深色科学观测站主题（CSS 自定义属性设计系统）
- 全中文界面，Noto Sans SC 字体
- 营销落地页 + 控制台分离架构
- 响应式布局，适配移动端

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/zxzok/brainnet.git
cd brainnet

# 安装依赖
pip install -e .

# 或者手动安装
pip install -r requirements.txt
```

### 2. 启动 Web 应用

```bash
# 使用入口命令
brainnet-web

# 或直接运行
python web_app.py
```

访问 `http://localhost:6525` 查看落地页，点击「进入控制台」进入分析界面。

### 3. 配置 AI 功能

进入控制台后，点击导航栏「设置」，输入你的 Anthropic API 密钥即可启用 AI 对话功能。

获取密钥：https://console.anthropic.com/settings/keys

### 4. 命令行使用

```bash
# 处理本地 BIDS 数据集
brainnet-cli /path/to/bids --subject 01 --task rest

# 从 OpenNeuro 下载并处理
brainnet-cli --openneuro-id ds000114 --subject 01 --task rest

# 查看分析计划（不执行）
brainnet-cli --describe-plan /path/to/bids --subject 01 --task rest
```

## 页面导航

| 页面 | 路径 | 说明 |
|------|------|------|
| 落地页 | `/` | 平台介绍与宣传主页 |
| 控制台 | `/console` | 数据概览仪表盘 |
| 被试管理 | `/patients` | 被试列表、添加/编辑/查看 |
| 数据中心 | `/data` | 已下载数据集与下载状态 |
| OpenNeuro | `/openneuro` | 搜索、浏览公共数据集 |
| 数据集浏览 | `/openneuro/<id>/browse` | 被试/会话/运行选择与分析 |
| 分析结果 | `/analysis` | 所有分析任务 |
| 计算特征 | `/features` | 所有影像的特征列表 |
| 特征详情 | `/features/<image_id>` | 连接矩阵热图、动态状态图、指标表 |
| 网络可视化 | `/network` | 交互式脑网络 3D 可视化 |
| AI 对话 | `/chat` | Claude 智能分析对话 |
| 设置 | `/settings` | API 密钥配置 |
| 系统状态 | `/system/status` | 依赖检查与安装指引 |

## 仓库结构

```
brainnet/
├── main.py                  # CLI 入口，流水线编排
├── web_app.py               # Flask Web 应用
├── web_chat.py              # AI 对话蓝图（Claude SSE 流式）
├── analysis_service.py      # 分析服务层
├── runtime.py               # 运行时配置
├── __init__.py              # 包初始化，延迟加载可选依赖
├── requirements.txt         # Python 依赖
│
├── data_management.py       # BIDS 数据集索引
├── openneuro_client.py      # OpenNeuro GraphQL API 客户端
├── preprocessing.py         # 预处理流水线
├── static_analysis.py       # 静态连接分析
├── dynamic_analysis.py      # 动态连接分析
├── visualization.py         # Plotly 报告生成
├── templates.py             # 脑图谱模板管理
│
├── templates/               # Jinja2 模板（全中文）
│   ├── landing.html         #   落地页（独立页面，动画背景）
│   ├── base.html            #   基础布局（深色主题设计系统）
│   ├── settings.html        #   API 密钥设置
│   ├── index.html           #   控制台仪表盘
│   └── ...                  #   其他 20+ 页面
│
├── agents/                  # 多智能体工作流配置
│   ├── prompts/             #   各任务 Agent 提示词
│   └── sequential_agents.yaml
│
├── tests/                   # 测试（47 个用例）
│   ├── test_dataset_index.py
│   ├── test_roi_labels.py
│   ├── test_entrypoints.py
│   └── ...
│
├── CLAUDE.md                # AI 代理指令
├── AGENTS.md                # 多智能体编排文档
└── WORKFLOW.md              # 工作流执行文档
```

## 分析流水线

```
BIDS 数据集 / NIfTI 文件
        │
        ▼
┌─────────────────┐
│   数据索引       │  DatasetIndex：发现被试、会话、运行
└───────┬─────────┘
        ▼
┌─────────────────┐
│   预处理         │  空间平滑 → 时间滤波 → 去噪回归 → ROI 提取
└───────┬─────────┘
        ▼
┌─────────────────┐     ┌──────────────────────┐
│  静态连接分析    │     │  动态连接分析          │
│                 │     │                      │
│ Pearson 相关    │     │ 滑动窗口 + K-means    │
│   → 连接矩阵    │     │ 或 HMM / CAP         │
│   → 图论指标    │     │   → 状态序列          │
│     度          │     │   → 占用率            │
│     聚类系数    │     │   → 驻留时间          │
│     全局效率    │     │   → 转移概率          │
│     模块度      │     │                      │
└───────┬─────────┘     └──────────┬───────────┘
        │                          │
        └────────────┬─────────────┘
                     ▼
           ┌──────────────────┐
           │  结果存储与可视化  │
           │                  │
           │ SQLite 特征数据库 │
           │ 连接矩阵热图      │
           │ 状态时间线图      │
           │ HTML 交互报告     │
           │ Claude AI 解读   │
           └──────────────────┘
```

## 数据库

SQLite 数据库（`instance/brainnet.db`），应用启动时自动创建：

| 表名 | 说明 |
|------|------|
| `patients` | 被试信息（姓名、年龄、性别、诊断） |
| `mri_images` | MRI 影像记录（路径、类型、描述） |
| `features` | 计算特征（名称、值、类型：static/dynamic/static_node） |
| `openneuro_datasets` | 已下载 OpenNeuro 数据集元信息 |
| `download_tasks` | 下载任务日志 |
| `user_settings` | 用户设置（API 密钥等） |

## 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `ANTHROPIC_API_KEY` | Claude AI 密钥（也可通过 Web 设置页面配置） | — |
| `FSLDIR` | FSL 安装路径（高级预处理） | — |
| `OPENNEURO_CACHE_DIR` | OpenNeuro 缓存目录 | `~/.cache/openneuro` |

参考 `.env.example` 文件获取完整配置示例。

## 测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行单个测试
python -m pytest tests/test_dataset_index.py -v

# 代码检查
python -m ruff check .
```

当前状态：47 个测试通过，1 个跳过（hmmlearn 可选依赖）。

## 系统要求

- **Python** 3.10+
- **操作系统** Linux / macOS / Windows
- **必需依赖** Flask, NumPy, SciPy, Pandas, nibabel, nilearn, scikit-learn, networkx
- **可选依赖** hmmlearn（HMM 分析）、plotly（交互报告）、datalad（大文件下载）
- **可选外部工具** FSL / ANTs / SPM（仅高级预处理需要）

## 许可证

本项目仅用于研究和教学目的。
