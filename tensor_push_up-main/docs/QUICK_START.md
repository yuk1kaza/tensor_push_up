# 快速开始指南

## 正确的运行目录

**重要**: 所有命令必须在项目根目录下运行，而不是在父目录。

```bash
# 正确
cd D:\Programs\VScode\tensor_push_up-main\tensor_push_up-main
python src/preprocess.py --input data/raw --output data/processed

# 错误
cd D:\Programs\VScode\tensor_push_up-main
python src/preprocess.py --input data/raw --output data/processed
```

## 常用命令

### 1. 数据预处理

```bash
cd tensor_push_up-main
python src/preprocess.py --input data/raw --output data/processed
```

### 2. 模型训练

```bash
cd tensor_push_up-main
python src/train.py --config configs/train.yaml
```

### 3. 模型评估

```bash
cd tensor_push_up-main
python src/evaluate.py --model models/checkpoints/best.keras --data-dir data/processed
```

### 4. 实时推理（摄像头）

```bash
cd tensor_push_up-main
python demo.py --mode count --source 0
```

### 5. 视频推理

```bash
cd tensor_push_up-main
python demo.py --mode count --source video.mp4
```

### 6. 姿态估计演示

```bash
cd tensor_push_up-main
python demo.py --mode pose --source 0
```

## 项目目录结构

```
tensor_push_up-main/           # 这是正确的工作目录
├── README.md
├── .docs/                # 文档目录
├── src/                  # 源代码
├── configs/              # 配置文件
├── data/                 # 数据目录
└── models/               # 模型目录
```

## PowerShell 快捷方式

如果使用 PowerShell，可以创建一个函数简化命令切换：

```powershell
function cd-tensor {
    cd "D:\Programs\VScode\tensor_push_up-main\tensor_push_up-main"
}

# 然后直接使用
cd-tensor
python src/preprocess.py ...
```

## VS Code 集成

在 VS Code 中打开 `tensor_push_up-main` 作为工作区根目录：

1. 打开 VS Code
2. 文件 -> 打开文件夹
3. 选择 `tensor_push_up-main`

这样在 VS Code 终端中运行的命令就都在正确的目录下了。
