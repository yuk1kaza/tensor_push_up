# Ubuntu WSL2 GPU 训练指南

这个项目可以在原生 Windows 上运行，但 TensorFlow `>= 2.11` 在 Windows
环境里通常会退回到 CPU-only。  
如果你希望使用 NVIDIA GPU 训练，推荐链路是：

`Windows NVIDIA 驱动 -> WSL2 -> Ubuntu -> Python 虚拟环境 -> Linux 下的 TensorFlow GPU`

本指南基于你当前项目路径编写：

```text
D:\Programs\VScode\tensor_push_up-main\tensor_push_up-main
```

在 Ubuntu WSL 中，这个路径对应为：

```bash
/mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
```

## 你需要准备的东西

- Windows 11，或支持 WSL2 的较新版本 Windows 10
- NVIDIA 显卡
- 支持 CUDA on WSL 的 Windows 侧 NVIDIA 驱动
- 通过 WSL2 安装好的 Ubuntu

## 一个重要原则

不要在 WSL 里额外安装 Linux 版 NVIDIA 显示驱动。  
WSL 的 GPU 计算依赖的是 Windows 侧驱动。

## 1. 先在 Windows 侧检查环境

打开 PowerShell，先确认 Windows 本机可以识别 GPU：

```powershell
nvidia-smi
```

如果这里都失败了，先修复 Windows 驱动，再继续折腾 WSL。

安装或更新 WSL：

```powershell
wsl --install -d Ubuntu
wsl --update
wsl -l -v
```

确保你的 Ubuntu 发行版版本号是 `2`。

如果已经装了 Ubuntu，但还是 WSL1：

```powershell
wsl --set-version Ubuntu 2
```

## 2. 打开 Ubuntu WSL

进入 Ubuntu 后，先更新基础包：

```bash
sudo apt update
sudo apt upgrade -y
```

建议顺手装一些常用工具：

```bash
sudo apt install -y python3-venv python3-pip build-essential git libgl1 libegl1 libgles2 libopengl0
```

这些图形运行库对 MediaPipe 很重要。  
如果缺少它们，预处理时可能会报类似：

```text
libGLESv2.so.2: cannot open shared object file
```

## 3. 进入项目目录

```bash
cd /mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
pwd
```

确认你已经从 Linux 侧进入项目目录。

## 4. 创建 Linux 专用虚拟环境

不要复用 Windows 下的 `.venv\Scripts\activate`。  
请在 Ubuntu 里新建一个 Linux 侧虚拟环境：

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python --version
```

如果你使用 VS Code，推荐直接用 `Remote - WSL` 打开仓库，然后把解释器切到：

```text
.venv-wsl/bin/python
```

## 5. 在 WSL 中安装 TensorFlow GPU 版

先升级打包工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

安装 TensorFlow 及其 CUDA 依赖：

```bash
python -m pip install "tensorflow[and-cuda]"
```

如果下载速度非常慢，不建议硬等。像你现在这种 `572 MB` 主包只跑到十几 `kB/s`
的情况，通常说明到 PyPI 的链路太慢了。更实用的做法是中断后改用镜像源重试。

例如临时使用清华镜像：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "tensorflow[and-cuda]"
```

如果你直接使用项目脚本，也可以这样：

```bash
bash scripts/setup_wsl_gpu.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

然后再安装项目其余依赖。

因为 `requirements.txt` 里也包含了 `tensorflow>=2.12.0`，最稳妥的方式是：

```bash
grep -v "^tensorflow" requirements.txt > /tmp/tensor_push_up_requirements_wsl.txt
python -m pip install -r /tmp/tensor_push_up_requirements_wsl.txt
```

如果当前环境里没有 `grep`，那就手工复制一份 requirements，并去掉 TensorFlow 那一行。

## 6. 验证 WSL 内是否能看到 GPU

先看 WSL 能不能访问 GPU：

```bash
nvidia-smi
```

如果 `nvidia-smi` 不在 PATH 里，可以试试：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

然后验证 TensorFlow：

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

预期结果：

- 至少出现一个类似 `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`
- 不能是空列表 `[]`

## 7. 训练前先确认项目数据是否可用

这个项目现在会在训练前主动检查数据质量。  
一个常见问题是：数据集中只有 `other` 一个类别。

先看是否已有处理好的数据：

```bash
ls data/processed
```

如果还没做预处理：

```bash
python src/preprocess.py --input data/raw --output data/processed
```

注意：如果 `data/labels` 里没有真实标注 JSON，预处理通常会把大部分样本默认标成 `other`，训练会被提前拦下。

## 8. 在 Ubuntu WSL 中开始训练

正常训练：

```bash
python src/train.py --config configs/train.yaml
```

短烟雾测试：

```bash
python src/train.py --config configs/train.yaml --epochs 1 --no-export
```

如果你只是想验证流程，允许单类别数据继续跑：

```bash
python src/train.py --config configs/train.yaml --epochs 1 --no-export --allow-single-class
```

## 9. 现在日志里会多输出什么

训练器现在会额外打印：

- 当前平台
- TensorFlow 版本
- 可见 GPU 列表
- 数据真实输入 shape
- 配置 shape 与真实数据不一致时的警告
- 类别缺失时的警告

正常的 WSL GPU 启动日志应该更像：

- `platform=linux`
- 至少有一个可见 GPU

而原生 Windows 下通常会是：

- `platform=win32`
- `gpus=[]`
- 同时提示你改用 WSL2 训练

## 10. 推荐的 VS Code 工作流

从 Windows 侧操作：

1. 安装 VS Code 的 `WSL` 扩展
2. 执行 `WSL: Connect to WSL`
3. 打开 `/mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main`
4. 选择 `.venv-wsl/bin/python` 作为解释器
5. 在 VS Code 的 WSL 终端里执行训练命令

这样可以避免把 Windows Python、Windows TensorFlow 和 Linux TensorFlow 混在一起。

## 11. 常见问题排查

### `tf.config.list_physical_devices('GPU')` 返回 `[]`

按这个顺序排查：

1. Windows 下 `nvidia-smi` 是否正常
2. `wsl -l -v` 是否确认 Ubuntu 运行在版本 `2`
3. Ubuntu WSL 内 `nvidia-smi` 是否正常
4. 你是不是在 Linux 虚拟环境里安装了 `tensorflow[and-cuda]`
5. 当前 `python` 是否确实来自 `.venv-wsl`

### 训练时报“Only one class is present”

这是数据问题，不是 GPU 问题。

重点检查：

- `data/labels/` 是否真的有 JSON 标注文件
- 加完标签后是否重新跑过预处理
- `data/processed/train_labels.npy` 是否还全是 `2`

### 配置里的输入 shape 与数据实际 shape 不一致

训练器现在会告警，并优先使用数据真实 shape。  
但从工程一致性上讲，你还是应该重新生成数据或同步更新配置。

### 我已经在 Windows 装了 CUDA Toolkit，还需要做什么

可以装，但这个项目是否能在 WSL 里用 GPU，关键仍然是：

- Windows 侧驱动正常
- WSL2 正常
- Ubuntu WSL 内能看到 GPU
- TensorFlow 在 Linux 虚拟环境里正确安装

不需要在 WSL 里再手工完整搭一遍 Linux 显卡驱动。

## 12. 最短可复制流程

如果你的 Windows 驱动和 WSL2 都已经是好的，最短流程如下：

```bash
cd /mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "tensorflow[and-cuda]"
grep -v "^tensorflow" requirements.txt > /tmp/tensor_push_up_requirements_wsl.txt
python -m pip install -r /tmp/tensor_push_up_requirements_wsl.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python src/train.py --config configs/train.yaml
```

## 13. 项目内已集成的脚本入口

如果你不想每一步都手敲命令，可以直接使用项目里已经准备好的脚本：

- `scripts/setup_wsl_gpu.sh`
  用于在 Ubuntu WSL 中创建 `.venv-wsl`、安装 TensorFlow GPU 和其余依赖
- `scripts/setup_wsl_gpu.ps1`
  用于从 Windows PowerShell 触发 WSL 内的环境初始化
- `scripts/wsl_tensorflow_env.sh`
  用于自动把 `.venv-wsl` 内 `nvidia/*/lib` 加入 `LD_LIBRARY_PATH`
- `scripts/verify_wsl_gpu.sh`
  用于验证 WSL 内 `nvidia-smi` 和 TensorFlow GPU 是否可用
- `scripts/train_wsl.sh`
  用于在 WSL 中执行烟雾测试或正式训练
- `scripts/train_wsl.ps1`
  用于从 Windows PowerShell 启动 WSL 训练流程

典型使用方式：

```bash
bash scripts/setup_wsl_gpu.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
bash scripts/verify_wsl_gpu.sh
bash scripts/train_wsl.sh --smoke
bash scripts/train_wsl.sh
```

说明：

- `verify_wsl_gpu.sh` 和 `train_wsl.sh` 现在会自动加载
  `scripts/wsl_tensorflow_env.sh`
- 这会把 `.venv-wsl` 中 pip 安装的 CUDA/cuDNN 动态库路径加入
  `LD_LIBRARY_PATH`
- 这样可以修复 `Cannot dlopen some GPU libraries` 这类问题

## 参考资料

- TensorFlow pip 安装文档：<https://www.tensorflow.org/install/pip>
- Microsoft WSL 安装文档：<https://learn.microsoft.com/windows/wsl/install>
- Microsoft WSL GPU 说明：<https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl>
- NVIDIA CUDA on WSL 文档：<https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
