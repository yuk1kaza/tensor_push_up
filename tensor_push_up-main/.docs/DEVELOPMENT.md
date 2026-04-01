# 开发文档

这份文档面向当前仓库的实际状态，重点说明：

- 代码结构
- 数据与标签组织方式
- 训练、推理、发布的开发流程
- 当前已知限制与建议

## 当前项目状态

当前项目已经具备以下能力：

- 三分类数据集：`pushup` / `jumping_jack` / `other`
- MediaPipe Tasks API 特征提取已修复，不再产生全零特征
- WSL2 GPU 训练链路可用
- 模型可完成训练、评估、导出
- 离线视频计数可正常输出结果视频
- 俯卧撑与开合跳计数器已根据当前视频集做过一轮放宽调优

## 目录说明

```text
tensor_push_up-main/
├── .docs/                       # 文档目录
├── configs/                     # 配置文件
├── data/
│   ├── raw/                     # 训练原始视频
│   ├── labels/                  # 标签 JSON
│   ├── processed/               # 预处理结果
│   ├── inference/               # 推理测试视频
│   └── external/                # 外部下载数据
├── logs/                        # 训练与评估日志
├── models/
│   ├── checkpoints/             # 训练模型
│   └── exported/                # 导出模型
├── release_assets/              # GitHub Release 打包产物
├── results/
│   └── inference/               # 推理输出视频
├── scripts/                     # 自动化脚本
├── src/                         # 核心源代码
├── demo.py                      # 演示入口
├── security_audit.py            # 安全审计脚本
├── test_model.py                # 模型烟雾测试
└── test_preprocess.py           # 预处理烟雾测试
```

## 核心模块

### `src/pose_estimator.py`

负责：

- MediaPipe Pose 关键点提取
- 关节角度计算
- 时序模型输入特征提取

说明：

- 当前训练能正常工作的前提之一，是这里的 Tasks API 关键点提取逻辑已经修复

### `src/preprocess.py`

负责：

- 扫描 `data/raw/`
- 读取 `data/labels/`
- 自动补全缺失标签
- 提取特征
- 切分训练/验证/测试数据

当前标签策略：

- 优先读取已有 JSON
- 如果 JSON 中没有条目，则优先按文件夹推断类别
- 文件名规则只作为兜底

### `src/model.py`

负责：

- LSTM / BiLSTM / CNN / Transformer 模型定义
- 模型构建、编译、回调
- checkpoint 加载
- SavedModel / H5 导出

当前注意点：

- 已兼容 Keras 3 的自定义类加载与导出

### `src/train.py`

负责：

- 加载 `data/processed/*.npy`
- 自动校验输入 shape
- 自动校验标签分布
- 训练、评估、导出

### `src/infer.py`

负责：

- 单视频推理
- 批量推理
- 摄像头推理
- 模型分类 + 计数器结合

当前说明：

- 离线视频模式默认输出视频文件，不弹实时窗口
- `--exercise auto` 已可用，但动作类型明确时，手动指定更稳定

### `src/counter.py`

负责：

- 俯卧撑状态机
- 开合跳状态机
- 动作计数逻辑

当前说明：

- 已经按当前数据集放宽阈值
- 更偏向“实用计数”，不是极严格的动作质量审核

## 当前默认阈值

配置位于：

- [`configs/train.yaml`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/configs/train.yaml)

### Push-up

- `high_angle = 140`
- `low_angle = 105`
- `stability_frames = 2`
- `cooldown_frames = 10`

### Jumping Jack

- `open_ankle_distance = 1.2`
- `closed_ankle_distance = 0.5`
- `stability_frames = 2`
- `cooldown_frames = 8`

## 数据与标签规范

训练数据推荐放在：

```text
data/raw/
  pushup/
  jumping_jack/
  other/
```

推理测试视频推荐放在：

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

## 开发工作流

### 1. 新增训练数据

1. 把视频放进 `data/raw/<class>/`
2. 运行标签生成脚本
3. 检查 `data/labels/*.json`

### 2. 重跑预处理

```bash
python src/preprocess.py --input data/raw --output data/processed --no-parallel
```

说明：

- 当前在本项目环境下，预处理更推荐直接在 Windows 侧执行
- 原因是 WSL 下 MediaPipe 在部分环境中可能存在 EGL / segmentation fault 风险

### 3. WSL2 烟雾训练

```bash
bash scripts/train_wsl.sh --venv .venv-wsl --smoke
```

### 4. WSL2 正式训练

```bash
bash scripts/train_wsl.sh --venv .venv-wsl
```

### 5. 离线推理

```powershell
python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise pushup --output results/inference/test_pushup_01_counted.mp4
```

### 6. 模型发布

```bash
python scripts/package_github_release.py --name tensor-push-up-model-YYYYMMDD
```

输出目录：

- `release_assets/`

## 自动化脚本

### 环境与训练

- `scripts/setup_wsl_gpu.sh`
- `scripts/setup_wsl_gpu.ps1`
- `scripts/verify_wsl_gpu.sh`
- `scripts/train_wsl.sh`
- `scripts/train_wsl.ps1`
- `scripts/wsl_tensorflow_env.sh`

### 数据与标签

- `scripts/generate_labels_from_filenames.py`
- `scripts/organize_raw_videos.py`
- `scripts/download_kaggle_dataset.py`

### 发布

- `scripts/package_github_release.py`

## 当前已知限制

### 1. 数据切分仍偏样本级

当前预处理产物仍然是窗口级样本切分，尚未完全升级为严格的视频级切分。

这意味着：

- 当前评估结果可能偏乐观

### 2. 计数规则仍需按数据集微调

当前阈值是基于现有数据集调出来的，不一定适合所有机位、动作幅度和人群。

### 3. `other` 类仍需要持续扩充

虽然项目已经支持三分类，但 `other` 类的多样性仍然直接影响推理阶段的拒识能力。

## 推荐开发优先级

如果继续迭代项目，建议优先做：

1. 更严格的视频级切分
2. 计数器调试信息增强
3. 有效动作 / 无效动作判定
4. 数据质量与标签质量抽检
5. 推理阶段更稳的 auto 路由

## 参考文档

- [`DOCS_INDEX.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/DOCS_INDEX.md)
- [`Data_Requirement.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/Data_Requirement.md)
- [`PROJECT_RUNTIME_FLOW_CN.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/PROJECT_RUNTIME_FLOW_CN.md)
- [`INFERENCE_USAGE_CN.md`](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/INFERENCE_USAGE_CN.md)
