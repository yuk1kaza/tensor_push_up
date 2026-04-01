# 特征工程落地计划

这份计划基于当前项目的最佳建议路线：

`成熟库（MediaPipe） + 特征工程 + 规则系统 + 状态机`

目标不是完全替换现有模型，而是在当前项目基础上增加一条更可解释、可维护、适合动作有效性判定的能力路径。

## 总体目标

在当前项目中增加一套面向动作有效性与计数判定的特征工程方案，用于支持：

- 动作分类辅助
- 有效动作 / 无效动作判定
- 是否计入计数
- 后续动作质量分析

## 总体路线

```text
视频 -> MediaPipe 姿态关键点 -> 几何/时序特征 -> 规则判断 / 传统模型 -> 状态机计数
```

## 阶段 1：统一目标与输出

### 目标

先明确这套特征工程路线优先解决什么问题：

1. 区分 `pushup` / `jumping_jack` / `other`
2. 判断某次动作是否有效
3. 决定某次动作是否应该计入计数

### 预期输出

每次动作周期最终输出类似：

```json
{
  "exercise": "pushup",
  "rep_index": 5,
  "is_valid": true,
  "should_count": true,
  "metrics": {
    "min_elbow_angle": 88.3,
    "max_elbow_angle": 156.2,
    "torso_deviation_max": 24.7
  }
}
```

## 阶段 2：整理特征层

### 当前已有基础

当前项目已经能提供：

- MediaPipe 关键点
- 肘、肩、髋、膝角度
- 脚距
- 部分规则计数逻辑

### 需要新增的特征

建议把特征拆成三层。

#### A. 单帧几何特征

- 左右肘角
- 左右肩角
- 左右髋角
- 左右膝角
- 左右踝距离
- 左右手腕相对肩位置
- 躯干主轴角
- 左右对称误差

#### B. 短窗口时序特征

- 肘角变化速度
- 脚距变化速度
- 肩角变化速度
- 最近 N 帧平均值 / 最大值 / 最小值

#### C. 单次动作周期统计特征

##### Push-up

- `min_elbow_angle`
- `max_elbow_angle`
- `elbow_angle_range`
- `torso_deviation_max`
- `knee_angle_min`
- `rep_duration`
- `left_right_elbow_gap_max`

##### Jumping Jack

- `max_ankle_distance`
- `min_ankle_distance`
- `arms_up_ratio`
- `max_shoulder_angle`
- `rep_duration`
- `left_right_sync_error`

## 阶段 3：新增模块设计

建议新增以下模块，而不是把逻辑全部塞回已有文件。

### `src/feature_engineering.py`

负责：

- 从关键点提取单帧几何特征
- 从短窗口提取时序特征
- 生成可供规则系统或传统模型使用的结构化特征

### `src/rep_analyzer.py`

负责：

- 结合状态机，识别一个完整动作周期
- 对单次动作周期提取统计特征
- 输出 `is_valid` / `should_count` / 指标字典

### `src/form_rules.py`

负责：

- 俯卧撑有效动作规则
- 开合跳有效动作规则
- 后续扩展更多动作质量规则

### 可选：`src/classical_model.py`

负责：

- 用传统模型做有效性判定或动作分类辅助
- 例如 LogisticRegression / RandomForest / XGBoost

## 阶段 4：先做规则版 MVP

### 原则

第一版不先训练“有效/无效模型”，而是先做规则版。

原因：

- 可解释性最强
- 能快速落地
- 与当前项目结构最契合
- 不额外依赖大规模新标注

### Push-up 规则第一版

示例规则：

- 完整周期必须存在：`HIGH -> LOW -> HIGH`
- `min_elbow_angle <= 105`
- `max_elbow_angle >= 140`
- `torso_deviation_max <= 45`
- `knee_angle_min >= 某阈值`

满足则：

- `is_valid = true`
- `should_count = true`

否则：

- `is_valid = false`
- `should_count = false`

### Jumping Jack 规则第一版

示例规则：

- 完整周期必须存在：`CLOSED -> OPEN -> CLOSED`
- `max_ankle_distance >= 1.2`
- `min_ankle_distance <= 0.5`
- `arms_up_ratio` 达到要求
- 手臂与脚部动作同步性达到要求

满足则计数，否则不计数。

## 阶段 5：把规则接入计数器

### 当前问题

当前 `counter.py` 主要是在状态机层面判断是否计数。

### 改造目标

让 `counter.py` 负责：

- 动作阶段切换
- 找到一次完整动作周期

让 `rep_analyzer.py` / `form_rules.py` 负责：

- 判定本次动作是否有效

### 最终流程

```text
动作状态机识别到一个完整周期
    -> 提取该周期统计特征
    -> 规则判定 valid / invalid
    -> valid 才加 1
```

## 阶段 6：加入训练型辅助判定（可选）

在规则版跑稳后，再考虑传统模型辅助。

### 可训练任务

#### 任务 1：动作类型分类

- `pushup`
- `jumping_jack`
- `other`

#### 任务 2：动作有效性分类

- `valid`
- `invalid`

### 推荐传统模型

- `LogisticRegression`：最简单 baseline
- `RandomForest`：可解释性较好
- `XGBoost / LightGBM`：适合结构化特征

## 阶段 7：数据标注增强

如果要支持更稳的有效性判断，建议逐步扩展标签字段：

```json
{
  "action_type": "pushup",
  "count": 0,
  "is_valid": false,
  "invalid_reason": [
    "elbow_not_low_enough"
  ]
}
```

### 推荐优先级

1. 先补 `is_valid`
2. 再补 `invalid_reason`
3. 最后再考虑更细的评分

## 阶段 8：验证方案

每阶段都建议做小规模验证。

### 验证 1：特征有效性

- 检查提取出的特征不是全 0
- 检查不同动作的统计特征分布有差异

### 验证 2：规则有效性

- 随机抽若干视频
- 人工核对计数结果
- 检查哪些动作被错误判成 invalid

### 验证 3：传统模型可分性

- 先用 LogisticRegression 或 RandomForest
- 看结构化特征是否能把 `valid / invalid` 分开

## 阶段 9：文档与接口

落地时建议同步更新：

- `README.md`
- `.docs/DEVELOPMENT.md`
- `.docs/API.md`
- `.docs/INFERENCE_USAGE_CN.md`

并说明：

- 当前是规则判定
- 哪些阈值可调
- 哪些输出用于解释动作有效性

## 推荐实施顺序

1. 新增 `feature_engineering.py`
2. 新增 `rep_analyzer.py`
3. 新增 `form_rules.py`
4. 先实现规则版 push-up 有效动作判定
5. 再实现规则版 jumping jack 有效动作判定
6. 把 `counter.py` 接入“valid 才计数”逻辑
7. 再决定是否增加传统模型辅助

## 当前最适合的下一步

如果按最稳妥、最有产出的方式推进，建议先做：

**第一版 Push-up / Jumping Jack 动作周期级特征提取与规则判定**

这一步成功后，你的项目就会从：

- “动作分类 + 计数”

升级成：

- “动作分类 + 有效动作判定 + 计数”
