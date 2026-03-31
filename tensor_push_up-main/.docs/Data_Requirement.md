# 数据集与标注要求

这份文档说明当前项目在采集、整理、标注和扩展数据时的推荐规范。

## 当前支持的类别

- `pushup`
- `jumping_jack`
- `other`

## 推荐目录结构

训练数据建议统一放在：

```text
data/raw/
  pushup/
  jumping_jack/
  other/
```

推理测试视频建议单独放在：

```text
data/inference/
  pushup/
  jumping_jack/
  other/
```

这样可以避免把推理视频误混入训练集。

## 类别定义建议

### pushup

- 标准俯卧撑
- 包含完整周期：高位 -> 低位 -> 高位
- 尽量避免把明显不相关动作放进该类

### jumping_jack

- 标准开合跳
- 包含完整周期：闭合 -> 张开 -> 闭合
- 不要混入其它体操或抬手类动作

### other

建议放入：

- TaiChi
- PullUps
- HandStandPushups
- WallPushups
- 站立
- 走动
- 过渡动作
- 不属于前两类的运动片段

目标是让模型学会在非目标动作出现时拒识。

## 标注文件

标签文件位于：

```text
data/labels/
```

当前使用的主要文件有：

- `pushup_dataset_labels.json`
- `jumping_jack_dataset_labels.json`
- `other_dataset_labels.json`

## 自动生成标签

项目已经支持自动根据目录或文件名补标签。

推荐命令：

```bash
python scripts/generate_labels_from_filenames.py --input data/raw --labels data/labels --overwrite
```

规则是：

1. 优先按视频所在文件夹判断类别
2. 如果 JSON 中还没有对应视频，就自动补入对应类别的 JSON 文件
3. 文件名规则只作为兜底

## 标注字段建议

当前 JSON 结构大致如下：

```json
{
  "example.mp4": {
    "action_type": "pushup",
    "count": null,
    "start_frame": 0,
    "end_frame": 299,
    "notes": "Auto-generated..."
  }
}
```

建议：

- `action_type`
  必须准确
- `count`
  能填就填，尤其是后续做有效动作判定时很有价值
- `start_frame` / `end_frame`
  尽量覆盖真正动作段，不要把太多准备帧和收尾帧混进去

## 数据质量建议

### 1. 覆盖多样性

每一类尽量覆盖：

- 不同人物
- 不同背景
- 不同服装
- 不同速度
- 不同光照
- 不同机位轻微变化

### 2. 类别平衡

类别不要差太多。

当前项目已经出现过：
- `pushup` 远多于 `jumping_jack`

这会导致模型偏向样本更多的类。

### 3. `other` 要多样

不要只放一种静止动作。

如果 `other` 太单一，模型会学到一个很窄的“非目标动作”概念。

## 当前计数与有效动作判定相关建议

项目当前已经对俯卧撑和开合跳做了更宽松的计数规则调优，以适配现有视频集。

如果你后续要做“这次动作是否应该计入次数”的判定，建议：

1. 保留 `action_type`
2. 尽量补 `count`
3. 后续可以扩展字段，例如：

```json
"is_valid": true,
"invalid_reason": []
```

## 使用流程

推荐顺序：

1. 把视频放进 `data/raw/<class>/`
2. 运行标签生成脚本
3. 运行预处理：

```bash
python src/preprocess.py --input data/raw --output data/processed --no-parallel
```

4. 在 WSL 中训练：

```bash
bash scripts/train_wsl.sh --venv .venv-wsl
```

5. 推理测试：

```bash
python src/infer.py --source data/inference/pushup/test_pushup_01.mp4 --model models/checkpoints/best_model.keras --exercise pushup --output results/inference/test_pushup_01_counted.mp4
```
