# API 文档

本文档提供了 Tensor Push Up 项目中主要模块的 API 接口说明。

## 目录

- [安全模块 (security.py)](#安全模块-securitypy)
- [姿态估计 (pose_estimator.py)](#姿态估计-pose_estimatorpy)
- [动作计数 (counter.py)](#动作计数-counterpy)
- [模型定义 (model.py)](#模型定义-modelpy)
- [工具函数 (utils.py)](#工具函数-utilspy)

---

## 安全模块 (security.py)

### 主要函数

#### validate_file_path(file_path: str, allowed_dirs: Optional[List[str]] = None) -> bool
验证文件路径是否安全，防止目录遍历攻击。

**参数：**
- `file_path`: 待验证的文件路径
- `allowed_dirs`: 允许的目录列表（默认为当前目录）

**返回：**
- `True` - 路径安全
- `False` - 路径不安全

**示例：**
```python
from src.security import validate_file_path

# 验证路径
if validate_file_path("./video.mp4"):
    print("路径安全")
else:
    print("路径不安全")
```

#### validate_file_size(file_path: str, max_size: int) -> bool
验证文件大小是否在允许范围内。

**参数：**
- `file_path`: 文件路径
- `max_size`: 最大允许字节数

**返回：**
- `True` - 文件大小允许
- `False` - 文件过大

#### validate_file_extension(file_path: str, allowed_extensions: Set[str]) -> bool
验证文件扩展名是否在允许列表中。

#### sanitize_filename(filename: str) -> str
清理文件名，移除危险字符。

#### validate_video_file(file_path: str) -> bool
验证视频文件的路径、大小和扩展名。

#### validate_model_file(file_path: str) -> bool
验证模型文件的路径、大小和扩展名。

#### validate_config_file(file_path: str) -> bool
验证配置文件的路径、大小和扩展名。

---

## 姿态估计 (pose_estimator.py)

### PoseEstimator 类

#### 初始化
```python
PoseEstimator(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    static_image_mode: bool = False,
    model_complexity: int = 1,
    model_path: Optional[str] = None
)
```

**参数：**
- `min_detection_confidence`: 最小检测置信度 (0.0-1.0)
- `min_tracking_confidence`: 最小跟踪置信度 (0.0-1.0)
- `static_image_mode`: 是否为静态图像模式
- `model_complexity`: 模型复杂度 (0=轻量, 1=中等, 2=重型)
- `model_path`: 自定义模型文件路径

#### 主要方法

##### process_frame(frame: np.ndarray, timestamp_ms: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[Dict]]
处理单帧图像并提取姿态。

**参数：**
- `frame`: BGR 格式的图像帧
- `timestamp_ms`: 视频模式下的时间戳

**返回：**
- `keypoints`: 关键点数组 (33, 4) 或 None
- `angles`: 关节角度字典或 None

**示例：**
```python
from src.pose_estimator import PoseEstimator
import cv2

estimator = PoseEstimator()
frame = cv2.imread("frame.jpg")
keypoints, angles = estimator.process_frame(frame)

if keypoints is not None:
    print(f"检测到姿态，肘部角度: {angles['left_elbow']}")
```

##### process_video(video_path: str, output_path: Optional[str] = None, visualize: bool = False) -> List[Tuple[np.ndarray, Dict]]
处理整个视频文件。

**参数：**
- `video_path`: 视频文件路径
- `output_path`: 可视化输出路径
- `visualize`: 是否生成可视化

**返回：**
- 每帧的 (keypoints, angles) 元组列表

##### visualize_pose(frame, keypoints, angles, thickness=2) -> np.ndarray
在图像上绘制姿态骨架和关节角度。

##### normalize_keypoints(keypoints, method="bounding_box") -> np.ndarray
标准化关键点以减少位置和尺度差异。

##### extract_features(keypoints, angles) -> np.ndarray
提取用于机器学习的特征向量（50维）。

#### 关键点索引
```python
LEFT_SHOULDER = 11   # 左肩
RIGHT_SHOULDER = 12  # 右肩
LEFT_ELBOW = 13      # 左肘
RIGHT_ELBOW = 14     # 右肘
LEFT_WRIST = 15      # 左腕
RIGHT_WRIST = 16     # 右腕
LEFT_HIP = 23        # 左髋
RIGHT_HIP = 24       # 右髋
LEFT_KNEE = 25       # 左膝
RIGHT_KNEE = 26      # 右膝
LEFT_ANKLE = 27      # 左踝
RIGHT_ANKLE = 28     # 右踝
```

---

## 动作计数 (counter.py)

### PushUpCounter 类

#### 初始化
```python
PushUpCounter(
    high_angle_threshold: float = 150.0,
    low_angle_threshold: float = 90.0,
    stability_frames: int = 3,
    cooldown_frames: int = 10,
    torso_angle_threshold: float = 30.0
)
```

**参数：**
- `high_angle_threshold`: 高位状态的肘部角度阈值
- `low_angle_threshold`: 低位状态的肘部角度阈值
- `stability_frames`: 状态转换所需的最小持续帧数
- `cooldown_frames`: 计数后冷却帧数
- `torso_angle_threshold`: 躯干最大偏转角度

#### 主要方法

##### process_frame(keypoints: Optional[np.ndarray], angles: Optional[Dict[str, float]], frame_idx: int = 0) -> Dict[str, any]
处理单帧并更新计数器状态。

**参数：**
- `keypoints`: 33个关键点的数组
- `angles`: 关节角度字典
- `frame_idx`: 当前帧索引

**返回：**
- 包含计数和状态信息的字典

**示例：**
```python
from src.counter import PushUpCounter
import numpy as np

counter = PushUpCounter()
keypoints = np.zeros((33, 4))
keypoints[:, 3] = 1.0  # 设置为可见

angles = {'left_elbow': 160, 'right_elbow': 160}

for i in range(100):
    result = counter.process_frame(keypoints, angles, i)
    print(f"帧 {i}: 计数 = {counter.count}")

print(f"总共完成 {counter.count} 个俯卧撑")
```

#### 属性
- `count`: 当前完成次数
- `state`: 当前状态 (PushUpState 枚举)
- `stability_buffer`: 稳定性缓冲区
- `current_cooldown`: 当前冷却帧数

### JumpingJackCounter 类

#### 初始化
```python
JumpingJackCounter(
    open_ankle_distance: float = 0.3,
    closed_ankle_distance: float = 0.1,
    wrist_shoulder_threshold: float = 0.05,
    stability_frames: int = 3,
    cooldown_frames: int = 10
)
```

#### 主要方法

##### process_frame(keypoints: Optional[np.ndarray], angles: Optional[Dict[str, float]], frame_idx: int = 0) -> Dict[str, any]
处理单帧并更新开合跳计数器。

**参数：**
- `keypoints`: 33个关键点的数组
- `angles`: 关节角度字典（可选）
- `frame_idx`: 当前帧索引

**返回：**
- 包含计数和状态信息的字典

---

## 模型定义 (model.py)

### 主要函数

#### create_model(input_shape: Tuple[int, int], num_classes: int = 3, model_type: str = "lstm", **kwargs) -> keras.Model
创建动作分类模型。

**参数：**
- `input_shape`: 输入形状 (window_size, features)
- `num_classes`: 输出类别数
- `model_type`: 模型类型 ("lstm", "bilstm", "cnn", "transformer")
- `**kwargs`: 额外模型参数

**返回：**
- Keras 模型对象

**示例：**
```python
from src.model import create_model

# 创建LSTM模型
model = create_model(
    input_shape=(30, 50),
    num_classes=3,
    model_type="lstm",
    lstm_units=[128, 64],
    dropout_rate=0.5
)

# 查看模型结构
model.summary()
```

#### compile_model(model: keras.Model, learning_rate: float = 0.001) -> keras.Model
编译模型。

**参数：**
- `model`: Keras 模型
- `learning_rate`: 学习率

#### load_model_from_checkpoint(checkpoint_path: str, custom_objects: Optional[Dict] = None) -> keras.Model
从检查点加载模型。

**参数：**
- `checkpoint_path`: 模型文件路径
- `custom_objects`: 自定义对象字典

**返回：**
- 加载的 Keras 模型

#### export_model(model: keras.Model, export_path: str, format: str = "saved_model")
导出模型。

**参数：**
- `model`: Keras 模型
- `export_path`: 导出路径
- `format`: 导出格式 ("saved_model", "h5", "tflite")

### ModelInference 类

用于推理的包装类，包含预测平滑功能。

#### 初始化
```python
ModelInference(
    model_path: str,
    window_size: int = 30,
    smoothing_window: int = 5,
    confidence_threshold: float = 0.6
)
```

#### 主要方法

##### predict(features: np.ndarray) -> Tuple[str, float]
预测动作类别。

**参数：**
- `features`: 特征数组 (window_size, num_features)

**返回：**
- `(action_class, confidence)`: 动作类别和置信度

##### update(features: np.ndarray) -> Tuple[str, float]
更新预测缓冲区并返回平滑后的结果。

##### reset()
重置预测缓冲区。

---

## 工具函数 (utils.py)

### 配置函数

#### load_config(config_path: str) -> Dict[str, Any]
加载 YAML 配置文件。

**参数：**
- `config_path`: 配置文件路径

**返回：**
- 配置字典

#### save_config(config: Dict[str, Any], config_path: str)
保存配置到 YAML 文件。

### 视频 I/O

#### VideoReader 类
视频读取器，支持上下文管理。

**初始化：**
```python
VideoReader(video_path: str)
```

**使用示例：**
```python
from src.utils import VideoReader

with VideoReader("video.mp4") as reader:
    for frame in reader:
        # 处理帧
        pass
```

**属性：**
- `fps`: 帧率
- `width`: 视频宽度
- `height`: 视频高度
- `frame_count`: 总帧数
- `duration`: 视频时长（秒）

#### VideoWriter 类
视频写入器。

**初始化：**
```python
VideoWriter(
    output_path: str,
    fps: int = 30,
    resolution: Optional[Tuple[int, int]] = None,
    fourcc: str = 'mp4v'
)
```

### 可视化函数

#### draw_pose_on_image(frame, keypoints, connections, **kwargs) -> np.ndarray
在图像上绘制姿态。

#### draw_text_overlay(frame, text_lines, **kwargs) -> np.ndarray
在图像上绘制文本覆盖层。

#### draw_counter_display(frame, pushup_count, jumping_jack_count, **kwargs) -> np.ndarray
绘制计数器显示。

### 计算函数

#### calculate_angle(point1, vertex, point2) -> float
计算三点形成的角度。

#### calculate_distance(point1, point2) -> float
计算两点之间的欧氏距离。

### 数据处理函数

#### normalize_keypoints(keypoints, reference_point=None, scale=None) -> np.ndarray
标准化关键点。

#### create_sliding_windows(sequence, window_size, stride=1) -> List[np.ndarray]
创建滑动窗口。

#### pad_sequence(sequence, target_length, **kwargs) -> np.ndarray
填充序列到目标长度。

#### split_train_val_test(data_list, **kwargs) -> Tuple[List, List, List]]
分割数据为训练、验证、测试集。

### 指标函数

#### calculate_count_metrics(predicted_counts, true_counts, tolerance=1) -> Dict[str, float]
计算基于计数的评估指标。

**返回的指标：**
- `mae`: 平均绝对误差
- `mape`: 平均百分比误差
- `count_accuracy`: 计数准确率
- `exact_accuracy`: 精确准确率

### 工具类

#### Timer
简单的计时器。

**使用示例：**
```python
from src.utils import Timer

with Timer() as timer:
    # 执行操作
    pass

print(f"耗时: {timer.elapsed():.2f}秒")
```

#### ProgressTracker
进度跟踪器，带 ETA 估算。

---

## 配置文件结构

`configs/train.yaml` 包含以下部分：

```yaml
# 模型配置
model:
  name: "lstm_mlp"
  input_shape: [30, 50]
  lstm_units: [128, 64]
  dropout_rate: 0.5
  num_classes: 3
  learning_rate: 0.001

# 数据配置
data:
  window_size: 30
  stride: 5
  train_split: 0.7
  val_split: 0.2
  test_split: 0.1
  batch_size: 32

# 训练配置
training:
  epochs: 100
  early_stopping_patience: 15
  use_tensorboard: true

# 动作检测阈值
action_thresholds:
  pushup:
    high_angle: 150.0
    low_angle: 90.0
    stability_frames: 3
    cooldown_frames: 10

# 置信度阈值
confidence:
  detection_threshold: 0.5
  classification_threshold: 0.6
  smoothing_window: 5
```

---

## 错误代码

| 错误 | 描述 | 解决方法 |
|------|------|---------|
| FileNotFoundError | 文件未找到 | 检查文件路径是否正确 |
| ValueError | 参数值无效 | 检查参数范围和类型 |
| RuntimeError | 运行时错误 | 查看详细错误信息 |
| SecurityError | 安全验证失败 | 检查文件路径和大小限制 |

---

## 版本信息

- Python: 3.8+
- TensorFlow: 2.12+
- MediaPipe: 0.10+
- OpenCV: 4.8+
