# Tensor Push Up

一个基于深度学习的人体动作识别与计数项目，当前聚焦三类动作：

- `pushup`
- `jumping_jack`
- `other`

项目目标是通过视频中的人体关键点与时序特征，完成：

- 动作分类
- 离线视频计数
- 实时/准实时推理
- 后续扩展到动作有效性判断

## 当前状态

当前项目已经完成并验证通过：

- WSL2 GPU 训练链路可用
- MediaPipe 特征提取已修复，不再产出全零特征
- 数据集已支持 `pushup` / `jumping_jack` / `other`
- 训练模型可成功保存、加载、导出
- 离线视频计数可正常输出结果视频
- 俯卧撑与开合跳计数器已根据当前视频集做过一轮放宽调优

## 推荐文档入口

完整文档现在统一放在：

- [`tensor_push_up-main/.docs/DOCS_INDEX.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/DOCS_INDEX.md)

重点文档：

- [`WSL_GPU_TRAINING_CN.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/WSL_GPU_TRAINING_CN.md)
- [`PROJECT_RUNTIME_FLOW_CN.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/PROJECT_RUNTIME_FLOW_CN.md)
- [`INFERENCE_USAGE_CN.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/INFERENCE_USAGE_CN.md)
- [`Data_Requirement.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/Data_Requirement.md)

## 项目目录

实际代码和数据位于内层目录 `tensor_push_up-main/`，当前推荐按下面的结构理解：

```text
tensor_push_up-main/
├── .docs/                       # 项目文档总目录
│   ├── DOCS_INDEX.md            # 文档索引
│   ├── README.md                # 项目概览
│   ├── DEVELOPMENT.md           # 开发说明
│   ├── API.md                   # API 参考
│   ├── QUICK_START.md           # 快速开始
│   ├── WSL_GPU_TRAINING*.md     # WSL2 GPU 训练说明
│   ├── INFERENCE_USAGE_CN.md    # 推理与计数使用说明
│   ├── PROJECT_RUNTIME_FLOW_CN.md
│   └── Data_Requirement.md      # 数据集与标注规范
├── configs/
│   └── train.yaml               # 训练、计数阈值和路径配置
├── data/
│   ├── raw/                     # 训练原始视频
│   │   ├── pushup/
│   │   ├── jumping_jack/
│   │   └── other/
│   ├── labels/                  # 标签 JSON
│   ├── processed/               # 预处理后的特征、样本与切分结果
│   ├── inference/               # 推理测试视频
│   │   ├── pushup/
│   │   ├── jumping_jack/
│   │   └── other/
│   └── external/                # 外部下载的数据集或参考数据
├── models/
│   ├── checkpoints/             # 训练得到的最佳模型和中间权重
│   └── exported/                # 导出后的 SavedModel / H5 模型
├── logs/                        # 数据集信息、训练历史、评估结果
├── results/
│   └── inference/               # 离线计数输出视频
├── release_assets/              # GitHub Release 打包产物
├── scripts/
│   ├── setup_wsl_gpu.*          # WSL2 GPU 环境初始化
│   ├── verify_wsl_gpu.sh        # WSL2 GPU 可见性验证
│   ├── train_wsl.*              # WSL2 训练入口
│   ├── generate_labels_from_filenames.py
│   ├── organize_raw_videos.py
│   ├── package_github_release.py
│   └── download_kaggle_dataset.py
├── src/
│   ├── pose_estimator.py        # 姿态估计与特征提取
│   ├── preprocess.py            # 数据预处理
│   ├── model.py                 # 模型定义、加载、导出
│   ├── train.py                 # 训练流程
│   ├── evaluate.py              # 模型评估
│   ├── infer.py                 # 离线/实时推理
│   ├── counter.py               # 动作计数状态机
│   ├── utils.py                 # 视频/配置/可视化工具
│   ├── security.py              # 安全校验
│   └── __init__.py
├── demo.py                      # 快速演示入口
├── security_audit.py            # 安全审计脚本
├── test_model.py                # 模型烟雾测试
├── test_preprocess.py           # 预处理烟雾测试
├── requirements.txt             # Python 依赖
├── pose_landmarker_*.task       # MediaPipe Pose 模型文件
└── .gitignore
```

关键目录职责：

- `.docs/`：所有文档的统一入口，建议从 `DOCS_INDEX.md` 开始阅读
- `data/raw/`：训练视频，按类别文件夹组织
- `data/labels/`：与训练视频对应的标签文件
- `data/processed/`：预处理后生成的 `.npy` 特征、标签、切分数据和元信息
- `data/inference/`：仅用于推理测试的视频，不应混入训练集
- `models/checkpoints/`：训练中得到的最佳模型，优先使用 `best_model.keras`
- `models/exported/`：导出的 SavedModel、H5 等部署产物
- `scripts/`：训练、数据整理、模型发布等自动化脚本
- `results/inference/`：离线计数后输出的视频结果
- `release_assets/`：准备上传 GitHub Release 的模型和说明文件

## 数据组织方式

训练数据推荐这样放：

```text
data/raw/
  pushup/
  jumping_jack/
  other/
```

推理测试视频推荐这样放：

```text
data/inference/
  pushup/
  jumping_jack/
  other/
```

标签生成命令：

```bash
python scripts/generate_labels_from_filenames.py --input data/raw --labels data/labels --overwrite
```

规则说明：

- 优先按文件夹判断类别
- 如果 JSON 中没有对应条目，就自动补入对应类别的标签文件
- 文件名规则只作为兜底

## 训练流程

### Windows / WSL2

推荐在 WSL2 中训练，不建议继续使用原生 Windows TensorFlow 做正式训练。

环境文档见：

- [`WSL_GPU_TRAINING_CN.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/WSL_GPU_TRAINING_CN.md)

### 预处理

在仓库根目录执行：

```bash
python src/preprocess.py --input data/raw --output data/processed --no-parallel
```

### WSL2 GPU 训练

```bash
bash scripts/train_wsl.sh --venv .venv-wsl
```

烟雾测试：

```bash
bash scripts/train_wsl.sh --venv .venv-wsl --smoke
```

## 模型使用

训练完成后，推荐使用：

- [`models/checkpoints/best_model.keras`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/models/checkpoints/best_model.keras)

离线视频计数示例：

```powershell
python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise pushup --output results/inference/test_pushup_01_counted.mp4
```

开合跳示例：

```powershell
python src/infer.py --source data/inference/jumping_jack/test_jj_01.mp4 --model models/checkpoints/best_model.keras --exercise jumping_jack --output results/inference/test_jj_01_counted.mp4
```

说明：

- 离线视频模式默认不会弹实时窗口，而是直接写出结果视频
- 输入视频尽量使用原始视频，不要重复拿 `*_counted.mp4` 再做推理
- 如果目标视频动作明确，优先手动指定 `--exercise`
- `demo.py` 与 `infer.py` 的参数习惯不同：
  - `demo.py` 打开摄像头时推荐使用 `--camera 0`
  - `infer.py` 打开摄像头时可以使用 `--source 0`

## 当前计数规则

为了更贴合当前视频集，项目当前使用的是放宽后的计数阈值。

### 俯卧撑

- `high_angle_threshold = 140`
- `low_angle_threshold = 105`
- `stability_frames = 2`
- `torso_angle_threshold = 45`

### 开合跳

- `open_ankle_threshold = 1.2`
- `closed_ankle_threshold = 0.5`
- `stability_frames = 2`
- `cooldown_frames = 8`

这意味着当前更偏向“实用计数”，而不是特别严格的标准动作审核。

## 模型发布

当前项目已支持打包 GitHub Release 资产：

```bash
python scripts/package_github_release.py --name tensor-push-up-model-YYYYMMDD
```

打包结果会输出到：

- `release_assets/`

## 当前已知限制

- 当前训练集切分仍偏样本级，不是完全严格的视频级切分
- 计数规则已经调优，但仍可能需要根据你自己的视频继续细调
- `infer.py` 的 `--exercise auto` 已可用，但如果目标动作明确，手动指定更稳

## 下一步可扩展方向

- 动作有效 / 无效判定
- 更严格的视频级切分
- 动作质量分析
- 更多动作类型
- 更稳的自动路由与自动计数
