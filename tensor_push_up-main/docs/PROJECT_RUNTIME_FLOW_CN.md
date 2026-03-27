# 项目运行流程图

这份文档描述的是当前项目在接入 Ubuntu WSL2 GPU 训练方案之后的推荐运行路径。

## 总体流程

```mermaid
flowchart TD
    A[Windows: NVIDIA 驱动 / WSL2] --> B[Ubuntu WSL2]
    B --> C[创建并激活 .venv-wsl]
    C --> D[安装 TensorFlow GPU 和项目依赖]
    D --> E[验证 GPU: verify_wsl_gpu.sh]
    E --> F[准备原始视频 data/raw]
    F --> G[准备标签 data/labels]
    G --> H[预处理 preprocess.py]
    H --> I[生成 data/processed/*.npy]
    I --> J[烟雾测试 train_wsl.sh --smoke]
    J --> K[正式训练 train_wsl.sh]
    K --> L[模型权重 models/checkpoints]
    K --> M[训练日志 logs]
    K --> N[导出模型 models/exported]
    L --> O[评估 evaluate.py]
    L --> P[推理 infer.py / demo.py]
```

## 环境流程

```mermaid
flowchart LR
    W1[Windows PowerShell] --> W2[wsl --install / wsl --update]
    W2 --> W3[nvidia-smi]
    W3 --> U1[Ubuntu WSL2]
    U1 --> U2[setup_wsl_gpu.sh]
    U2 --> U3[.venv-wsl]
    U3 --> U4[tensorflow and-cuda]
    U4 --> U5[verify_wsl_gpu.sh]
```

## 数据与训练流程

```mermaid
flowchart TD
    R1[data/raw: 原始视频] --> P1[preprocess.py]
    R2[data/labels: 标签 JSON] --> P1
    P1 --> P2[pose_estimator.py 提取关键点与角度]
    P2 --> P3[切片成时序窗口]
    P3 --> P4[data/processed/train val test]
    P4 --> T1[train.py]
    T1 --> T2[读取真实 input shape]
    T2 --> T3[检查类别分布]
    T3 --> T4[训练模型]
    T4 --> T5[best_model.keras]
```

## 推理与使用流程

```mermaid
flowchart TD
    M1[best_model.keras] --> I1[infer.py]
    I1 --> I2[pose_estimator.py]
    I2 --> I3[提取时序特征]
    I3 --> I4[model.py 分类]
    I4 --> I5[counter.py 状态机计数]
    I5 --> I6[demo.py / 实时显示 / 视频输出]
```

## 推荐执行顺序

1. Windows 侧确认 `nvidia-smi` 正常
2. Ubuntu WSL2 中运行 `bash scripts/setup_wsl_gpu.sh`
3. 运行 `bash scripts/verify_wsl_gpu.sh`
4. 准备 `data/raw` 和 `data/labels`
5. 执行 `python src/preprocess.py --input data/raw --output data/processed`
6. 先跑 `bash scripts/train_wsl.sh --smoke`
7. 再跑 `bash scripts/train_wsl.sh`
8. 训练后使用 `python src/evaluate.py ...` 做评估
9. 使用 `demo.py` 或 `infer.py` 进行推理和演示

## 关键目录

- `data/raw/`：原始视频
- `data/labels/`：标注 JSON
- `data/processed/`：预处理后的训练数据
- `models/checkpoints/`：训练中保存的模型
- `models/exported/`：导出后的模型
- `logs/`：训练日志
- `scripts/`：WSL GPU 训练辅助脚本
- `docs/`：训练说明、执行计划、流程图

## 最常用命令

```bash
source .venv-wsl/bin/activate
bash scripts/verify_wsl_gpu.sh
python src/preprocess.py --input data/raw --output data/processed
bash scripts/train_wsl.sh --smoke
bash scripts/train_wsl.sh
```

## 当前数据现状提醒

当前仓库里的数据已经推进到双类别：

- [pushup_dataset_labels.json](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/data/labels/pushup_dataset_labels.json)
- [jumping_jack_dataset_labels.json](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/data/labels/jumping_jack_dataset_labels.json)

当前 `data/processed` 的标签分布已经是：

- `pushup`
- `jumping_jack`

这意味着现在已经可以做有意义的双类别训练。  
如果你后续想要完整支持三分类，还需要再补充 `other` 类视频和标签。
