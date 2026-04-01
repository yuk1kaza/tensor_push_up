# 训练后模型计数使用说明

## 目标视频应该放在哪里

如果你只是想用训练好的模型去测试一个视频、做动作识别或计数，
不建议把它放进 `data/raw/`。

推荐放到：

```text
data/inference/
  pushup/
  jumping_jack/
  other/
```

这样可以和训练数据目录分开，避免误把测试视频混入训练集。

## 推荐放置方式

例如：

```text
data/inference/pushup/test_pushup_01.mp4
data/inference/jumping_jack/test_jj_01.mp4
data/inference/other/test_other_01.mp4
```

注意：

- `--source` 必须写完整文件名
- 必须包含文件扩展名，例如 `.mp4` / `.avi`

正确：

```text
data/inference/jumping_jack/jackjump1.mp4
```

错误：

```text
data/inference/jumping_jack/jackjump1
```

## 一个重要提醒

请尽量使用原始输入视频，不要把之前已经导出过可视化叠字的
`*_counted.mp4` 再拿来重复推理。

推荐：

- `test_pushup_01.mp4`

不推荐：

- `test_pushup_01_counted.mp4`

因为已经叠加过骨架、文字和计数框的视频，会干扰新的姿态估计。

## 如何使用训练好的模型去计数

### 俯卧撑视频

```powershell
python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise pushup --output results/inference/test_pushup_01_counted.mp4
```

### 开合跳视频

```powershell
python src/infer.py --source data/inference/jumping_jack/test_jj_01.mp4 --model models/checkpoints/best_model.keras --exercise jumping_jack --output results/inference/test_jj_01_counted.mp4
```

### 自动识别动作类型

```powershell
python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise auto --output results/inference/test_pushup_01_counted.mp4
```

## PowerShell 与 Bash 的区别

如果你在 Windows PowerShell 中运行命令：

- 不要使用 Bash 风格的 `\` 换行
- 最稳妥的方式是整行写完

正确示例：

```powershell
python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise pushup --output results/inference/test_pushup_01_counted.mp4
```

如果你确实想换行，PowerShell 要用反引号 `` ` ``，不是反斜杠 `\`。

## 在 WSL 中运行

如果你希望在 WSL 环境中跑推理，可以这样：

```powershell
wsl -d Ubuntu bash -lc "cd /mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main && source scripts/wsl_tensorflow_env.sh && activate_tensor_push_up_wsl_env .venv-wsl && python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise pushup --output results/inference/test_pushup_01_counted.mp4"
```

## 输出结果

计数完成后，你会得到一个带可视化和计数结果的视频，例如：

```text
results/inference/test_pushup_01_counted.mp4
```

离线视频模式默认不会弹出实时窗口，而是直接写出结果视频。

## 批量处理一个文件夹中的视频

如果你想把一个目录中的视频全部都跑一遍，可以使用：

```powershell
python src/infer.py --batch-dir data/inference/jumping_jack --model models/checkpoints/best_model.keras --exercise jumping_jack --output-dir results/inference/jumping_jack_batch
```

如果目录里动作混合，也可以使用：

```powershell
python src/infer.py --batch-dir data/inference --model models/checkpoints/best_model.keras --exercise auto --output-dir results/inference/batch_auto
```

说明：

- 现在 `--batch-dir` 模式已经可以单独使用
- 不再错误地强制要求 `--source`
- 批量模式会逐个视频处理，并在 `--output-dir` 下写出结果视频与 `batch_results.json`

## 当前建议

如果目标视频里动作类型是明确的，优先使用：

- `--exercise pushup`
- `--exercise jumping_jack`

这样通常比 `--exercise auto` 更稳定。

## 当前计数规则说明

为了适配当前视频集，项目已经对计数规则做过一轮放宽：

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

这意味着当前输出结果更偏向“实用计数”，而不是“特别严格的标准动作审核”。
