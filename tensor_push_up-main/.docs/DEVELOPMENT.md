# 开发指南

## 项目状态

**当前版本:** 1.0.0
**状态:** 已就绪，可用于开发和测试

## 已实现功能

### 核心模块

| 模块 | 状态 | 说明 |
|------|------|------|
| 安全模块 (security.py) | ✅ 完成 | 路径验证、文件大小限制、扩展名验证 |
| 姿态估计 (pose_estimator.py) | ✅ 完成 | MediaPipe Tasks API，自动下载模型 |
| 动作计数 (counter.py) | ✅ 完成 | 俯卧撑和开合跳计数器 |
| 模型定义 (model.py) | ✅ 完成 | LSTM、Bi-LSTM、CNN、Transformer 模型 |
| 训练脚本 (train.py) | ✅ 完成 | 完整训练流程 |
| 推理脚本 (infer.py) | ✅ 完成 | 摄像头和视频推理 |
| 预处理 (preprocess.py) | ✅ 完成 | 视频数据预处理 |
| 评估脚本 (evaluate.py) | ✅ 完成 | 模型评估功能 |
| 工具函数 (utils.py) | ✅ 完成 | 视频 I/O、可视化、工具函数 |

### 安全特性

1. **路径验证** - 防止目录遍历攻击
2. **文件大小限制** - 防止 DoS 攻击
3. **扩展名验证** - 仅允许特定文件类型
4. **文件名清理** - 防止路径注入

### 已知问题

#### 1. MediaPipe 模型下载
**问题:** 首次运行需要下载模型文件

**解决:**
- 模型会自动从 Google Cloud 下载
- 需要 20-50 MB 的下载量
- 下载的模型文件保存在项目根目录

#### 2. MediaPipe 运行模式警告
**问题:** Tasks API 在非静态模式下可能显示警告

**影响:** 不影响功能

**解决:**
- 警告可以忽略
- 或使用 `static_image_mode=True` 进行单帧处理

#### 3. GPU 支持
**问题:** TensorFlow 2.11+ 在原生 Windows 上不支持 GPU

**解决:**
- 使用 WSL2
- 或使用 TensorFlow-DirectML 插件
- CPU 训练仍可工作（较慢）

## 开发环境设置

### 1. 克隆项目

```bash
git clone <repository-url>
cd tensor_push_up
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python -c "import tensorflow; import mediapipe; import cv2; print('依赖安装成功')"
```

## 开发工作流

### 添加新功能

1. **修改或创建模块**
   - 在 `src/` 目录下修改或创建文件
   - 遵循现有代码风格
   - 添加文档字符串

2. **更新配置**
   - 如需新参数，更新 `configs/train.yaml`

3. **编写测试**
   - 创建单元测试验证新功能
   - 测试边界情况和错误处理

4. **更新文档**
   - 更新 API.md 添加新接口
   - 更新 README.md 添加新功能说明

### 调试技巧

#### 1. 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

#### 2. 保存中间结果

```python
np.save('debug_keypoints.npy', keypoints)
np.save('debug_angles.npy', angles)
```

#### 3. 可视化特征

```python
import matplotlib.pyplot as plt
plt.plot(features)
plt.show()
```

## 代码风格指南

### 命名规范

- **类名:** PascalCase (如 `PoseEstimator`)
- **函数/方法名:** snake_case (如 `process_frame`)
- **常量:** UPPER_SNAKE_CASE (如 `MAX_VIDEO_SIZE`)
- **私有方法:** 前缀下划线 (如 `_calculate_angles`)

### 文档字符串

所有公共函数和类应包含文档字符串：

```python
def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """
    Process a single video frame and extract pose keypoints.

    Args:
        frame: Input image frame (BGR format)

    Returns:
        Tuple of (keypoints, angles)
    """
```

### 类型提示

使用类型提示提高代码可读性：

```python
from typing import Tuple, Optional, List, Dict

def function_name(
    param1: str,
    param2: Optional[int] = None
) -> Dict[str, any]:
    pass
```

## 性能优化建议

### 1. 姿态估计
- 使用轻量模型 (`model_complexity=0`)
- 减小视频分辨率进行预处理
- 批量处理多帧

### 2. 模型训练
- 使用适当的批量大小
- 启用数据并行（如可用 GPU）
- 使用混合精度训练

### 3. 推理
- 启用模型量化
- 批量推理多帧
- 使用时间窗口平滑减少抖动

## 测试指南

### 单元测试

```python
import unittest
from src.security import validate_file_path

class TestSecurity(unittest.TestCase):
    def test_validate_file_path(self):
        self.assertTrue(validate_file_path('./test.mp4'))
        self.assertFalse(validate_file_path('/etc/passwd'))
```

### 集成测试

运行完整流程测试：

```bash
# 1. 测试姿态估计
python demo.py --mode pose --source test_video.mp4

# 2. 测试计数
python demo.py --mode count --source test_video.mp4

# 3. 运行安全审计
python security_audit.py
```

## 常用命令

```bash
# 代码检查
black src/
isort src/

# 运行测试
python -m pytest tests/

# 训练模型
python src/train.py --config configs/train.yaml

# 评估模型
python src/evaluate.py --model models/checkpoints/best.keras

# 安全审计
python security_audit.py
```

## 贡献指南

### 提交 Pull Request

1. Fork 项目
2. 创建功能分支: `git checkout -b feature/amazing-feature`
3. 提交更改: `git commit -m 'Add amazing feature'`
4. 推送分支: `git push origin feature/amazing-feature`
5. 创建 Pull Request

### 报告 Bug

在 GitHub Issues 中报告 bug，包含：
- 问题描述
- 重现步骤
- 系统环境信息
- 错误日志

## 参考资料

- [MediaPipe 文档](https://developers.google.com/mediapipe)
- [TensorFlow 文档](https://www.tensorflow.org/)
- [OpenCV 文档](https://docs.opencv.org/)
- [CLAUDE.md](./CLAUDE.md) - Claude Code 开发指南
- [API.md](./API.md) - API 接口文档
