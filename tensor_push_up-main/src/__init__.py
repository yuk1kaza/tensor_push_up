"""
Tensor Push Up - Deep Learning-based Action Recognition and Counting

This package provides tools for:
- Pose estimation using MediaPipe
- Action classification using LSTM models
- Exercise counting with state machines
- Real-time and offline inference
"""

__version__ = "1.0.0"

from .pose_estimator import PoseEstimator, BatchPoseEstimator
from .counter import (
    PushUpCounter,
    JumpingJackCounter,
    ExerciseCounter,
    create_counter
)
from .model import (
    ActionClassifier,
    BidirectionalActionClassifier,
    TemporalCNN,
    TransformerClassifier,
    create_model,
    compile_model,
    ModelInference,
    ACTION_CLASSES,
    ACTION_LABELS
)
from .utils import (
    setup_logging,
    load_config,
    save_config,
    VideoReader,
    VideoWriter,
    draw_pose_on_image,
    draw_counter_display,
    Timer,
    ProgressTracker
)

__all__ = [
    'PoseEstimator',
    'BatchPoseEstimator',
    'PushUpCounter',
    'JumpingJackCounter',
    'ExerciseCounter',
    'create_counter',
    'ActionClassifier',
    'BidirectionalActionClassifier',
    'TemporalCNN',
    'TransformerClassifier',
    'create_model',
    'compile_model',
    'ModelInference',
    'ACTION_CLASSES',
    'ACTION_LABELS',
    'setup_logging',
    'load_config',
    'save_config',
    'VideoReader',
    'VideoWriter',
    'draw_pose_on_image',
    'draw_counter_display',
    'Timer',
    'ProgressTracker'
]
