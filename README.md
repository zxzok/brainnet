# BrainNet — 功能性磁共振成像分析平台

BrainNet 是一个模块化的 fMRI（功能性磁共振成像）预处理与分析平台，提供从数据获取、预处理、静态/动态功能连接分析到交互式报告生成的完整工作流。系统包含一个基于 Flask 的 Web 界面，支持患者管理、OpenNeuro 数据集浏览下载、MRI 图像上传、自动化分析和结果可视化。

## 功能特性

- **数据管理** — BIDS 格式数据集索引，OpenNeuro 数据集搜索、下载与浏览
- **预处理** — 空间平滑、时间滤波、运动校正、去噪回归、ROI 时间序列提取
- **静态连接分析** — Pearson 相关连接矩阵，图论指标（度、聚类系数、全局效率、模块度）
- **动态连接分析** — 滑动窗口分析，K-means 聚类 / 高斯 HMM / 共激活模式（CAP）
- **可视化** — 连接矩阵热图、动态状态时间线、状态占用率图表、交互式 HTML 报告
- **Web 界面** — 患者 CRUD、图像上传、后台分析、特征管理、数据集浏览器
- **LLM 解读** — 可选的 OpenAI / 本地 Transformer 模型自然语言结果解读

## 快速开始

### 1. 安装依赖

```bash
# 基础依赖（Web 界面 + 数据管理）
pip install Flask==2.3.3 requests openneuro-py

# 分析依赖（预处理 + 连接分析）
pip install numpy scipy pandas nibabel nilearn scikit-learn networkx

# 可选依赖
pip install plotly        # 交互式报告
pip install hmmlearn      # 隐马尔可夫模型分析
pip install openai        # LLM 结果解读
pip install datalad       # 大文件断点续传下载
```

或者一键安装全部：

```bash
pip install -r requirements.txt
pip install nibabel nilearn scikit-learn plotly hmmlearn
```

### 2. 启动 Web 应用

```bash
python web_app.py
```

访问 `http://localhost:6525` 即可使用。

### 3. 命令行使用

```bash
# 处理本地 BIDS 数据集
python main.py /path/to/bids_dataset --subject 01 --task rest

# 从 OpenNeuro 下载并处理
python main.py --openneuro-id ds000114 --subject 01 --task rest

# 为数据库中的患者生成报告
python main.py --patient-id 1
```

## 仓库结构

```
brainnet/
├── main.py                     # CLI 入口，流水线编排
├── web_app.py                  # Flask Web 应用（患者管理、分析、报告）
├── __init__.py                 # 包初始化，优雅处理可选依赖导入
├── requirements.txt            # Python 依赖列表
├── brainnet.db                 # SQLite 数据库（自动创建）
│
├── 数据模块
│   ├── data_management.py      # BIDS 数据集索引（DatasetIndex, BIDSFile, DatasetManager）
│   └── openneuro_client.py     # OpenNeuro GraphQL API 客户端
│
├── 预处理模块
│   ├── preprocessing.py        # 简化版预处理流水线（Preprocessor）
│   └── preprocessing_full.py   # 模块化预处理，可配置步骤（PreprocessPipeline）
│
├── 静态分析模块
│   ├── static_analysis.py      # StaticAnalyzer 封装（兼容旧接口）
│   └── static/                 # 模块化静态连接包
│       ├── connectivity.py     #   ConnectivityMatrix, Pearson 相关计算
│       ├── metrics.py          #   GraphMetrics, 度/聚类/效率/模块度
│       └── analyzer.py         #   StaticAnalyzer 编排类
│
├── 动态分析模块
│   ├── dynamic_analysis.py     # 动态分析封装（兼容旧接口）
│   └── dynamic/                # 模块化动态连接包
│       ├── config.py           #   DynamicConfig 配置
│       ├── analyzer.py         #   DynamicAnalyzer 编排类
│       ├── model.py            #   DynamicStateModel, DynamicMetrics
│       ├── window.py           #   滑动窗口计算
│       ├── kmeans.py           #   K-means 状态识别
│       ├── hmm.py              #   高斯 HMM 分析
│       ├── cap.py              #   共激活模式分析
│       ├── metrics.py          #   时间统计（占用率、驻留时间、转移概率）
│       ├── state_features.py   #   状态级图论指标
│       └── io.py               #   结果保存/加载
│
├── 其他模块
│   ├── visualization.py        # Plotly HTML 报告生成
│   ├── llm_interpretation.py   # LLM 自然语言结果解读
│   ├── templates.py            # 脑图谱模板管理（AAL, Harvard-Oxford, Schaefer）
│   └── multimodal.py           # 多模态数据处理
│
├── templates/                  # Flask HTML 模板（Jinja2）
│   ├── base.html               #   基础布局（导航栏、Bootstrap 5）
│   ├── index.html              #   首页
│   ├── patients.html           #   患者列表
│   ├── patient_detail.html     #   患者详情
│   ├── add_patient.html        #   添加患者
│   ├── edit_patient.html       #   编辑患者
│   ├── data_hub.html           #   数据中心
│   ├── openneuro.html          #   OpenNeuro 数据集列表
│   ├── openneuro_detail.html   #   数据集详情
│   ├── dataset_browse.html     #   数据集内容浏览
│   ├── features_detail.html    #   特征详情与可视化
│   ├── system_status.html      #   系统依赖状态
│   └── ...                     #   其他页面
│
└── tests/                      # 单元测试
    ├── test_dataset_index.py   #   BIDS 索引测试
    ├── test_openneuro_client.py#   OpenNeuro 客户端测试
    ├── test_roi_labels.py      #   ROI 标签测试
    ├── test_static_metrics.py  #   静态指标测试
    ├── test_dynamic_io.py      #   动态分析 I/O 测试
    └── test_hmm_auto_n_states.py#  HMM 自动状态数测试
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
           │ LLM 自然语言解读  │
           └──────────────────┘
```

## Web 界面

### 页面导航

| 页面 | 路径 | 说明 |
|------|------|------|
| 首页 | `/` | 患者列表总览 |
| 患者列表 | `/patients` | 所有患者 |
| 添加患者 | `/add_patient` | 新建患者记录 |
| 患者详情 | `/patient/<id>` | 患者信息、MRI 图像、分析操作 |
| 数据中心 | `/data` | 已下载数据集、下载状态 |
| OpenNeuro 浏览 | `/openneuro` | 搜索和浏览 OpenNeuro 数据集 |
| 数据集详情 | `/openneuro/<id>` | 数据集元信息、下载/浏览操作 |
| 数据集内容 | `/openneuro/<id>/browse` | 被试/会话/运行浏览，选择分析 |
| 特征详情 | `/features/<image_id>` | 分析结果可视化：热图、状态图、指标表 |
| 系统状态 | `/system/status` | 依赖检查与安装指引 |
| 生成报告 | `/patient/<id>/report` | 下载 HTML 分析报告 |

### API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/patients` | GET | 获取所有患者（JSON） |
| `/api/patients/<id>` | GET | 获取单个患者详情 |
| `/api/search?q=...` | GET | 搜索患者 |
| `/api/features/<image_id>/export` | GET | 导出特征为 JSON |
| `/api/download_status/<dataset_id>` | GET | 查询下载进度 |
| `/api/network_data` | GET | 获取网络可视化数据 |

### 核心操作流程

1. **添加患者** → `/add_patient` 填写患者信息
2. **获取数据** → 数据中心 → OpenNeuro 搜索 → 下载数据集
3. **浏览数据** → 数据集浏览 → 展开被试/会话 → 选择功能运行
4. **运行分析** → 选择患者 + 选择运行 → 后台执行预处理和分析
5. **查看结果** → 患者详情 → Features → 连接矩阵热图 + 动态状态图
6. **导出报告** → 生成 HTML 报告 / 导出 JSON 特征

## 编程接口

### 数据集索引

```python
from data_management import DatasetIndex

# 索引 BIDS 数据集（支持 func、anat、dwi）
index = DatasetIndex('/path/to/bids', datatypes=['func', 'anat', 'dwi'])

# 列出被试和运行
subjects = index.list_subjects()
runs = index.get_functional_runs(subjects[0])
for run in runs:
    print(f"被试: {run.subject}, 任务: {run.task}, 路径: {run.path}")

# 数据集统计
summary = index.summary()
print(f"被试数: {summary['subjects']}, 功能运行数: {summary['functional_runs']}")
```

### 预处理

```python
from preprocessing_full import (
    PreprocessPipeline, PreprocessPipelineConfig,
    SmoothingConfig, TemporalFilterConfig, RoiExtractionConfig,
)

# 配置预处理步骤
config = PreprocessPipelineConfig(
    smoothing=SmoothingConfig(enabled=True, fwhm=6.0),
    temporal_filter=TemporalFilterConfig(enabled=True, low_cut=0.01, high_cut=0.1),
    roi_extraction=RoiExtractionConfig(enabled=True),
)

pipeline = PreprocessPipeline(config)
outputs = pipeline.run('func.nii.gz')

roi_ts = outputs['roi_timeseries']      # (T, N_ROI) 数组
labels = outputs['roi_labels']          # ROI 标签列表
qc = outputs['qc_metrics']             # {'tSNR': ...}
```

### 静态连接分析

```python
from static_analysis import StaticAnalyzer

analyzer = StaticAnalyzer(threshold=0.2)
conn_matrix = analyzer.compute_connectivity(roi_ts, labels)
metrics = analyzer.compute_graph_metrics(conn_matrix)

print(f"全局效率: {metrics.global_metrics['global_efficiency']:.4f}")
print(f"模块度: {metrics.global_metrics['modularity']:.4f}")
print(f"平均最短路径: {metrics.global_metrics['average_shortest_path_length']:.4f}")
```

### 动态连接分析

```python
from dynamic import DynamicConfig, DynamicAnalyzer

# K-means 聚类
cfg = DynamicConfig(window_length=30, step=5, n_states=4, method='kmeans')
model = DynamicAnalyzer(cfg).analyse(roi_ts)

print(f"状态序列: {model.state_sequence}")
print(f"占用率: {model.metrics.occupancy}")
print(f"驻留时间: {model.metrics.dwell_time}")
print(f"转移概率矩阵:\n{model.metrics.transition_probs}")

# 自动选择最优状态数
cfg_auto = DynamicConfig(window_length=30, step=5, auto_n_states=True)
model_auto = DynamicAnalyzer(cfg_auto).analyse(roi_ts)

# HMM 分析
cfg_hmm = DynamicConfig(
    window_length=30, step=5, method='hmm',
    auto_n_states=True, n_states_criterion='bic',
)
model_hmm = DynamicAnalyzer(cfg_hmm).analyse(roi_ts)

# 共激活模式（CAP）
cfg_cap = DynamicConfig(
    window_length=30, step=5, method='cap', cap_threshold=1.5,
)
model_cap = DynamicAnalyzer(cfg_cap).analyse(roi_ts)
```

### 报告生成

```python
from visualization import ReportConfig, ReportGenerator

rep_gen = ReportGenerator(ReportConfig(output_dir='reports'))
report_path = rep_gen.generate(
    subject_id='sub-01',
    conn_matrix=conn_matrix,
    graph_metrics=metrics,
    dyn_model=model,
    roi_labels=labels,
    qc_metrics=qc,
    patient_info={"Name": "张三", "Age": 30},
)
print(f"报告已保存至: {report_path}")
```

## 预处理配置

所有预处理步骤通过 dataclass 配置，可按需启用/禁用：

| 步骤 | 配置类 | 默认状态 | 说明 |
|------|--------|----------|------|
| 切片时间校正 | `SliceTimingConfig` | 启用，method='none' | 支持 FSL、SPM |
| 运动校正 | `MotionCorrectionConfig` | 启用，method='none' | 支持 FSL、SPM、simple |
| 空间标准化 | `SpatialNormalizationConfig` | 禁用 | 支持 FSL、SPM、ANTs |
| 空间平滑 | `SmoothingConfig` | 禁用 | 高斯滤波，FWHM 可配置 |
| 时间滤波 | `TemporalFilterConfig` | 禁用 | Butterworth 带通滤波 |
| 去噪回归 | `NuisanceRegressionConfig` | 禁用 | OLS 回归去除混杂信号 |
| ROI 提取 | `RoiExtractionConfig` | 禁用 | 需要 atlas 脑图谱 |

## 外部工具

部分高级预处理步骤需要外部神经影像工具（非 Python 包）：

### FSL

```bash
# 安装后设置环境变量
export FSLDIR=/usr/local/fsl
source $FSLDIR/etc/fslconf/fsl.sh
export PATH="$FSLDIR/bin:$PATH"
```

提供命令：`slicetimer`（切片时间校正）、`mcflirt`（运动校正）、`flirt`（空间标准化）

### ANTs

```bash
# 通过 conda 安装
conda install -c conda-forge ants
```

提供命令：`antsRegistration`、`antsApplyTransforms`

### SPM

需要 MATLAB 或 Octave 环境，并将 SPM 目录加入 MATLAB 路径。

## 数据库

Web 应用使用 SQLite 数据库 `brainnet.db`，包含以下表：

| 表名 | 说明 |
|------|------|
| `patients` | 患者信息（ID、姓名、年龄、性别、诊断） |
| `mri_images` | MRI 图像记录（路径、类型、描述、所属患者） |
| `features` | 计算特征（特征名、值、类型：static/dynamic/static_node） |
| `openneuro_datasets` | 已下载 OpenNeuro 数据集（ID、元信息、路径、状态） |
| `download_tasks` | 下载任务日志（状态、错误信息） |

数据库在应用启动时自动创建，支持增量 schema 迁移。

## 环境变量

| 变量 | 用途 | 默认值 |
|------|------|--------|
| `FSLDIR` | FSL 安装路径 | — |
| `OPENNEURO_CACHE_DIR` | OpenNeuro 缓存目录 | `~/.cache/openneuro` |
| `OPENAI_API_KEY` | LLM 解读功能（或写入 `~/.openai_api_key`） | — |

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行单个测试
pytest tests/test_static_metrics.py -v
```

测试使用 `monkeypatch`、`tmp_path`、`@pytest.mark.parametrize` 和 `pytest.importorskip` 等 pytest 特性。

## 系统要求

- **Python** 3.10+
- **操作系统** Linux / macOS / Windows
- **必需 Python 包** 见 `requirements.txt`
- **可选外部工具** FSL、ANTs、SPM（仅高级预处理需要）

## 常见问题

**Q: Web 应用启动后提示分析功能不可用？**

A: 访问 `/system/status` 查看缺少哪些依赖，按提示安装即可。Web 应用的数据浏览和患者管理功能不需要分析依赖也能正常使用。

**Q: 如何在没有 FSL/ANTs 的环境下使用？**

A: 预处理步骤默认使用 `method='none'`（直通模式），不依赖外部工具。基本的平滑、滤波和 ROI 提取只需要 Python 包。

**Q: 动态分析报错 "window_length exceeds number of time points"？**

A: Web 界面已实现自适应参数（自动根据数据长度调整窗口大小）。API 使用时请确保 `window_length < 数据时间点数`。

**Q: 如何使用自定义脑图谱？**

A: 在 `RoiExtractionConfig` 中指定 `atlas_path` 参数指向自己的 NIfTI 图谱文件；或使用 `PreprocConfig` 的 `roi_templates` 参数加载内置图谱（`'aal'`、`'harvardoxford'`、`'schaefer'`）。

## 许可证

本项目仅用于研究和教学目的。
