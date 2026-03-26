# Tensor Push Up

一个基于 Tensor 模型的人体动作计数项目，用于识别并统计：

- 俯卧撑（Push-up）
- 开合跳（Jumping Jack）

项目目标是通过视频流（摄像头或本地视频）提取人体关键点/时序特征，训练动作分类与状态判定模型，并最终实现稳定、实时的动作次数统计。

## 功能特性

- 支持俯卧撑与开合跳两类动作识别
- 支持训练数据的采集、清洗、训练与评估
- 支持实时推理（WebCam）和离线推理（视频文件）
- 基于动作状态机进行计数，减少抖动导致的误计数
- 可扩展到更多健身动作（如深蹲、开合跳等）

## 技术思路

推荐采用以下两阶段方案：

1. 姿态估计阶段
	- 从每一帧提取人体关键点（如肩、肘、髋、膝、踝等）
	- 将关键点标准化，构造成时序特征

2. 动作识别与计数阶段
	- 使用 Tensor 模型（例如 MLP/LSTM/1D-CNN/Transformer）预测动作类别或动作阶段
	- 使用状态机规则进行一次完整动作的“起始 -> 结束”闭环计数

## 计数核心逻辑（建议）

### 俯卧撑计数

- 定义身体低位状态：肘角度小于某阈值，且躯干接近水平
- 定义身体高位状态：肘角度大于某阈值
- 当状态从高位 -> 低位 -> 高位完成一次闭环，计数 +1

### 开合跳计数

- 通过手臂和腿部开合动作判定
- 监测手臂上举至水平以上 + 双脚跳开的状态
- 监测手臂下放至身体两侧 + 双脚合拢的状态
- 完整开合闭环后计数 +1

### 抗抖动策略

- 预测结果滑动窗口平滑（如最近 N 帧多数投票）
- 状态切换设置最短持续帧数
- 设置动作冷却帧，避免同一次动作重复计数

## 项目结构（建议）

可参考如下目录组织：

```text
tensor_push_up/
├── README.md
├── data/
│   ├── raw/                 # 原始视频或关键点数据
│   ├── processed/           # 预处理后的训练数据
│   └── labels/              # 标注文件
├── models/
│   ├── checkpoints/         # 训练权重
│   └── exported/            # 导出模型（onnx/tflite/saved_model）
├── src/
│   ├── preprocess.py        # 数据预处理
│   ├── train.py             # 训练入口
│   ├── evaluate.py          # 评估脚本
│   ├── infer.py             # 推理脚本
│   ├── counter.py           # 动作计数状态机
│   └── utils.py             # 通用工具
├── requirements.txt
└── configs/
	 └── train.yaml           # 训练配置
```

> 当前仓库结构较简洁，你可以按上面结构逐步补齐。

## 环境准备

### 1. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果你还没有 `requirements.txt`，建议至少包含：

- tensorflow 或 pytorch（二选一）
- opencv-python
- numpy
- pandas
- scikit-learn
- matplotlib

## 数据准备

1. 采集俯卧撑和开合跳视频，保证：
	- 机位相对固定
	- 光照稳定
	- 覆盖不同身高、体型、速度
2. 标注数据：
	- 动作类别（pushup / jumping_jack）
	- 动作阶段（可选，用于更稳定计数）
3. 划分训练集、验证集、测试集（例如 7:2:1）

## 训练流程（示例）

```bash
python src/preprocess.py --input data/raw --output data/processed
python src/train.py --config configs/train.yaml
python src/evaluate.py --model models/checkpoints/best.pt --data data/processed
```

## 推理与计数（示例）

### 摄像头实时计数

```bash
python src/infer.py --source 0 --model models/checkpoints/best.pt --task count
```

### 视频文件计数

```bash
python src/infer.py --source demo.mp4 --model models/checkpoints/best.pt --task count
```

推理输出建议包含：

- 当前动作类别
- 当前动作阶段
- 俯卧撑累计次数
- 开合跳累计次数
- 当前帧置信度

## 评估指标（建议）

- 分类指标：Accuracy、Precision、Recall、F1
- 计数指标：
  - MAE（绝对误差）
  - MAPE（相对误差）
  - Count Accuracy（计数准确率）
- 实时性指标：FPS、端到端延迟

## 常见问题

### 1. 计数抖动严重

- 增加平滑窗口长度
- 提高状态切换阈值
- 引入阶段判别模型而非单帧分类

### 2. 侧身或遮挡时识别不稳定

- 增加多视角训练样本
- 做关键点缺失补全
- 引入时序模型提升鲁棒性

### 3. 不同速度下误差较大

- 扩充慢速/快速样本
- 训练时引入时间拉伸增强

## 后续优化方向

- 多人场景下的目标跟踪与独立计数
- 边缘设备部署（TFLite / TensorRT）
- 动作质量评分（标准度分析）
- 训练日志与可视化看板（TensorBoard / W&B）

## 贡献

欢迎提交 Issue 和 PR 来完善：

- 数据处理流程
- 模型结构与训练策略
- 计数稳定性与实时性能

## 许可证

可在此处补充项目许可证信息（如 MIT）。