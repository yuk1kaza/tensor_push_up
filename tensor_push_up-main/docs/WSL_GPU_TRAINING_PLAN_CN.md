# Ubuntu WSL2 GPU 训练执行计划

这是一份基于 [WSL_GPU_TRAINING_CN.md](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/docs/WSL_GPU_TRAINING_CN.md) 的落地计划。

目标：

- 在 Ubuntu WSL2 中完成 TensorFlow GPU 环境搭建
- 在 Linux 虚拟环境中运行本项目训练
- 确认训练真正使用 GPU
- 避免继续在 Windows 原生 TensorFlow 下做 CPU-only 训练

## 项目内已实现的入口

这份 plan 对应的项目内落地入口已经加入仓库：

- `scripts/setup_wsl_gpu.sh`
- `scripts/setup_wsl_gpu.ps1`
- `scripts/wsl_tensorflow_env.sh`
- `scripts/verify_wsl_gpu.sh`
- `scripts/train_wsl.sh`
- `scripts/train_wsl.ps1`
- `docs/WSL_GPU_TRAINING_CN.md`
- `docs/WSL_GPU_TRAINING.md`

建议你执行 plan 时，优先用这些入口，不要完全从零手敲。

补充：

- `verify_wsl_gpu.sh` 和 `train_wsl.sh` 已经内置了
  `LD_LIBRARY_PATH` 自动修复逻辑
- 如果 `tensorflow[and-cuda]` 已经装好，但 TensorFlow 仍提示
  `Cannot dlopen some GPU libraries`
  ，优先使用项目脚本重试，而不是手动反复 export

## 阶段 1：Windows 侧前置检查

- [ ] 在 PowerShell 中运行 `nvidia-smi`
- [ ] 确认显卡驱动正常，能看到 NVIDIA GPU
- [ ] 运行 `wsl -l -v`
- [ ] 确认 Ubuntu 已安装，且版本为 `2`
- [ ] 如果不是 WSL2，运行 `wsl --set-version Ubuntu 2`
- [ ] 运行 `wsl --update`

完成标准：

- Windows 能识别 GPU
- Ubuntu 运行在 WSL2

## 阶段 2：进入 Ubuntu WSL 并准备基础环境

- [ ] 打开 Ubuntu
- [ ] 执行 `sudo apt update`
- [ ] 执行 `sudo apt upgrade -y`
- [ ] 安装基础工具：

```bash
sudo apt install -y python3-venv python3-pip build-essential git libgl1 libegl1 libgles2 libopengl0
```

完成标准：

- Ubuntu 内 Python、venv、pip 可正常使用
- MediaPipe 依赖的 `libGLESv2.so.2` 等系统库已经安装

## 阶段 3：进入项目目录

- [ ] 在 Ubuntu 中进入项目目录：

```bash
cd /mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
pwd
```

- [ ] 确认当前目录正确

完成标准：

- 当前工作目录是 `/mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main`

## 阶段 4：创建 Linux 专用虚拟环境

- [ ] 创建虚拟环境：

```bash
python3 -m venv .venv-wsl
```

- [ ] 激活虚拟环境：

```bash
source .venv-wsl/bin/activate
```

- [ ] 确认 `python --version`
- [ ] 确认当前 `python` 来自 `.venv-wsl`

可选检查：

```bash
which python
```

完成标准：

- 使用的是 Ubuntu/Linux 侧虚拟环境，不是 Windows Python

## 阶段 5：安装 TensorFlow GPU 与项目依赖

- [ ] 升级打包工具：

```bash
python -m pip install --upgrade pip setuptools wheel
```

- [ ] 安装 TensorFlow GPU：

```bash
python -m pip install "tensorflow[and-cuda]"
```

- [ ] 如果下载速度异常慢，改用镜像源重试，例如：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "tensorflow[and-cuda]"
```

- [ ] 或者直接使用项目脚本：

```bash
bash scripts/setup_wsl_gpu.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

- [ ] 安装其余依赖：

```bash
grep -v "^tensorflow" requirements.txt > /tmp/tensor_push_up_requirements_wsl.txt
python -m pip install -r /tmp/tensor_push_up_requirements_wsl.txt
```

完成标准：

- TensorFlow 和项目依赖都安装完成
- 没有把 Windows 侧环境和 Linux 侧环境混用
- 如果走镜像安装，下载速度明显优于默认 PyPI

## 阶段 6：验证 WSL 内 GPU 可见

- [ ] 在 Ubuntu 中运行：

```bash
nvidia-smi
```

- [ ] 如果失败，尝试：

```bash
/usr/lib/wsl/lib/nvidia-smi
```

- [ ] 验证 TensorFlow GPU：

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

完成标准：

- `nvidia-smi` 在 WSL 中可用
- TensorFlow 至少看到一个 GPU 设备

## 阶段 7：检查项目数据是否可训练

- [ ] 查看 `data/processed` 是否存在
- [ ] 如果没有处理后的数据，执行：

```bash
python src/preprocess.py --input data/raw --output data/processed
```

- [ ] 检查 `data/labels/` 是否已有真实标注 JSON
- [ ] 确认不是所有标签都变成 `other`

完成标准：

- 训练数据已经准备好
- 至少不止一个类别

备注：

- 当前训练器已经会自动拦截“只有一个类别”的数据集
- 如果只想做流程烟雾测试，可以临时使用 `--allow-single-class`

## 阶段 8：进行一次短训练验证

- [ ] 执行 1 轮短训练：

```bash
python src/train.py --config configs/train.yaml --epochs 1 --no-export
```

- [ ] 观察日志中是否出现：
  - `platform=linux`
  - 非空 GPU 列表
  - 实际输入 shape

完成标准：

- 训练可以在 Ubuntu WSL 中启动
- 日志能看到 GPU

## 阶段 9：正式训练

- [ ] 执行正式训练：

```bash
python src/train.py --config configs/train.yaml
```

- [ ] 观察 checkpoint 是否正常写入 `models/checkpoints`
- [ ] 观察日志是否存在 shape 警告或类别缺失警告

完成标准：

- 模型正常训练
- 训练输出与日志完整

## 阶段 10：训练后检查

- [ ] 检查 `logs/` 是否生成训练日志
- [ ] 检查 `models/checkpoints/` 是否生成 `best_model.keras`
- [ ] 如启用导出，检查 `models/exported/`
- [ ] 如果训练效果异常，优先检查标签分布而不是先怀疑 GPU

完成标准：

- 训练产物完整
- 可以继续进入评估或推理阶段

## 推荐执行顺序

1. 先完成 Windows 驱动和 WSL2 检查
2. 再创建 Ubuntu 虚拟环境并安装 TensorFlow GPU
3. 先做 GPU 可见性验证
4. 再检查训练数据是否真的有多类别
5. 先跑 `--epochs 1` 的短验证
6. 最后再跑正式训练

## 最短行动版

如果你只是想快速推进，按下面顺序执行：

1. Windows PowerShell 里确认 `nvidia-smi`
2. `wsl -l -v` 确认 Ubuntu 是 WSL2
3. Ubuntu 中创建 `.venv-wsl`
4. 安装 `tensorflow[and-cuda]`
5. 用 `tf.config.list_physical_devices('GPU')` 验证 GPU
6. 检查 `data/labels` 和 `data/processed`
7. 跑 `python src/train.py --config configs/train.yaml --epochs 1 --no-export`
8. 没问题后再正式训练

## 风险提醒

- 不要在 Windows PowerShell 里继续做正式 GPU 训练
- 不要在 WSL 里额外安装 Linux 显卡驱动
- 不要混用 Windows 虚拟环境和 WSL 虚拟环境
- 不要在只有 `other` 类别的数据上做正式训练
