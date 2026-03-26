"""
Utility Functions Module

This module provides common utility functions for the Tensor Push Up project,
including video I/O, visualization, configuration loading, and logging.
"""


import os
import yaml
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime

# Import security functions
from .security import (
    validate_file_path, validate_file_size, validate_file_extension,
    sanitize_filename, validate_video_file, validate_model_file,
    validate_config_file, MAX_VIDEO_SIZE, MAX_MODEL_SIZE,
    ALLOWED_VIDEO_EXTENSIONS, ALLOWED_CONFIG_EXTENSIONS
)


# Configure logging
def setup_logging(
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_file: Specific log file name (auto-generated if None)

    Returns:
        Configured logger instance
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{log_dir}/tensor_push_up_{timestamp}.log"

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file with security validation.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Validate configuration file path
    if not validate_file_path(config_path):
        raise ValueError(f"Invalid configuration file path: {config_path}")

    # Validate file extension
    if not validate_file_extension(config_path, ALLOWED_CONFIG_EXTENSIONS):
        raise ValueError(f"Invalid configuration file extension: {config_path}")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Check file size (config files should be small)
    if not validate_file_size(config_path, 1024 * 1024):  # 1MB max
        raise ValueError(f"Configuration file too large: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    logger = logging.getLogger(__name__)
    logger.info(f"Configuration loaded from {config_path}")
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file with security validation.

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    # Validate configuration file path
    if not validate_file_path(config_path):
        raise ValueError(f"Invalid configuration file path: {config_path}")

    # Validate file extension
    if not validate_file_extension(config_path, ALLOWED_CONFIG_EXTENSIONS):
        raise ValueError(f"Invalid configuration file extension: {config_path}")

    # Sanitize output filename if it's in the current directory
    if os.path.dirname(config_path) == os.getcwd():
        filename = os.path.basename(config_path)
        sanitized = sanitize_filename(filename)
        if sanitized != filename:
            config_path = os.path.join(os.path.dirname(config_path), sanitized)

    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger = logging.getLogger(__name__)
    logger.info(f"Configuration saved to {config_path}")


# Video I/O Functions
class VideoReader:
    """
    Wrapper for video reading with automatic resource management.
    """

    def __init__(self, video_path: str):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file
        """
        # Validate video file path
        if not validate_file_path(video_path):
            raise ValueError(f"Invalid video file path: {video_path}")

        # Validate video file extension
        if not validate_file_extension(video_path, ALLOWED_VIDEO_EXTENSIONS):
            raise ValueError(f"Invalid video file extension: {video_path}")

        # Validate video file size
        if not validate_file_size(video_path, MAX_VIDEO_SIZE):
            raise ValueError(f"Video file too large: {video_path}")

        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0

        logger = logging.getLogger(__name__)
        logger.info(f"Video opened: {video_path}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")
        logger.info(f"  Frames: {self.frame_count}")
        logger.info(f"  Duration: {self.duration:.2f}s")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read next frame from video.

        Returns:
            Frame as numpy array or None if end of video
        """
        ret, frame = self.cap.read()
        return frame if ret else None

    def get_frame_at(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get frame at specific index.

        Args:
            frame_idx: Frame index to retrieve

        Returns:
            Frame as numpy array or None if invalid index
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        return self.read_frame()

    def __iter__(self):
        """Iterator for frames."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return self

    def __next__(self):
        """Get next frame."""
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Release video resources."""
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


class VideoWriter:
    """
    Wrapper for video writing with automatic resource management.
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        resolution: Optional[Tuple[int, int]] = None,
        fourcc: str = 'mp4v'
    ):
        """
        Initialize video writer.

        Args:
            output_path: Path to output video file
            fps: Frames per second
            resolution: Video resolution (width, height)
            fourcc: Video codec (mp4v, x264, etc.)
        """
        # Validate output file path
        if not validate_file_path(output_path):
            raise ValueError(f"Invalid output file path: {output_path}")

        # Validate video file extension
        if not validate_file_extension(output_path, ALLOWED_VIDEO_EXTENSIONS):
            raise ValueError(f"Invalid video file extension: {output_path}")

        # Sanitize output filename if it's in the current directory
        if os.path.dirname(output_path) == os.getcwd():
            filename = os.path.basename(output_path)
            sanitized = sanitize_filename(filename)
            if sanitized != filename:
                output_path = os.path.join(os.path.dirname(output_path), sanitized)

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

        self.output_path = output_path
        self.fps = fps
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)

        if resolution is None:
            # Default resolution
            self.width = 1280
            self.height = 720
        else:
            self.width, self.height = resolution

        self.writer = cv2.VideoWriter(
            output_path,
            self.fourcc,
            fps,
            (self.width, self.height)
        )

        if not self.writer.isOpened():
            raise ValueError(f"Failed to create video writer: {output_path}")

        logger = logging.getLogger(__name__)
        logger.info(f"Video writer initialized: {output_path}")
        logger.info(f"  Resolution: {self.width}x{self.height}")
        logger.info(f"  FPS: {self.fps}")

    def write_frame(self, frame: np.ndarray):
        """
        Write frame to video.

        Args:
            frame: Frame to write
        """
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        self.writer.write(frame)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def close(self):
        """Release video resources."""
        if hasattr(self, 'writer') and self.writer.isOpened():
            self.writer.release()
            logger = logging.getLogger(__name__)
            logger.info(f"Video saved: {self.output_path}")

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


# Visualization Functions
def draw_pose_on_image(
    frame: np.ndarray,
    keypoints: np.ndarray,
    connections: List[Tuple[int, int]],
    keypoint_color: Tuple[int, int, int] = (0, 255, 0),
    connection_color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
    radius: int = 3
) -> np.ndarray:
    """
    Draw pose keypoints and connections on image.

    Args:
        frame: Input image
        keypoints: Keypoints array (N, 2) or (N, 4)
        connections: List of (idx1, idx2) connection pairs
        keypoint_color: RGB color for keypoints
        connection_color: RGB color for connections
        thickness: Line thickness
        radius: Keypoint circle radius

    Returns:
        Image with pose visualization
    """
    vis_frame = frame.copy()
    height, width = frame.shape[:2]

    # Normalize keypoints if needed
    if keypoints.shape[1] == 4:
        # MediaPipe format with (x, y, z, visibility)
        xy = keypoints[:, :2]
        visibility = keypoints[:, 3]
    else:
        xy = keypoints
        visibility = np.ones(len(keypoints))

    # Draw connections
    for idx1, idx2 in connections:
        if visibility[idx1] > 0.5 and visibility[idx2] > 0.5:
            pt1 = (int(xy[idx1, 0] * width), int(xy[idx1, 1] * height))
            pt2 = (int(xy[idx2, 0] * width), int(xy[idx2, 1] * height))
            cv2.line(vis_frame, pt1, pt2, connection_color, thickness)

    # Draw keypoints
    for i, (x, y) in enumerate(xy):
        if visibility[i] > 0.5:
            pt = (int(x * width), int(y * height))
            cv2.circle(vis_frame, pt, radius, keypoint_color, -1)

    return vis_frame


def draw_text_overlay(
    frame: np.ndarray,
    text_lines: List[Tuple[str, Tuple[int, int, int]]],
    font_scale: float = 0.7,
    thickness: int = 2,
    margin: int = 10,
    background: bool = True
) -> np.ndarray:
    """
    Draw text overlay on frame.

    Args:
        frame: Input image
        text_lines: List of (text, color) tuples
        font_scale: Font size scale
        thickness: Text thickness
        margin: Margin from edges
        background: Whether to draw background behind text

    Returns:
        Image with text overlay
    """
    vis_frame = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = 30

    y_offset = margin + line_height

    for text, color in text_lines:
        if background:
            # Draw background rectangle
            (text_width, text_height), _ = cv2.getTextSize(
                text, font, font_scale, thickness
            )
            cv2.rectangle(
                vis_frame,
                (margin, y_offset - text_height - 5),
                (margin + text_width + 10, y_offset + 5),
                (0, 0, 0),
                -1
            )

        cv2.putText(
            vis_frame,
            text,
            (margin + 5, y_offset),
            font,
            font_scale,
            color,
            thickness
        )
        y_offset += line_height

    return vis_frame


def draw_counter_display(
    frame: np.ndarray,
    pushup_count: int = 0,
    jumping_jack_count: int = 0,
    action_type: str = "None",
    confidence: float = 0.0,
    fps: float = 0.0
) -> np.ndarray:
    """
    Draw counter display overlay on frame.

    Args:
        frame: Input image
        pushup_count: Current push-up count
        jumping_jack_count: Current jumping jack count
        action_type: Detected action type
        confidence: Detection confidence
        fps: Current FPS

    Returns:
        Image with counter display
    """
    text_lines = [
        (f"Push-ups: {pushup_count}", (0, 255, 255)),
        (f"Jumping Jacks: {jumping_jack_count}", (0, 255, 255)),
        (f"Action: {action_type}", (255, 0, 255)),
        (f"Confidence: {confidence:.2f}", (255, 255, 0)),
        (f"FPS: {fps:.1f}", (0, 255, 0))
    ]

    return draw_text_overlay(frame, text_lines)


# Angle Calculation Functions
def calculate_angle(
    point1: np.ndarray,
    vertex: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    Calculate angle at vertex formed by three points.

    Args:
        point1: First point
        vertex: Vertex point
        point2: Second point

    Returns:
        Angle in degrees (0-180)
    """
    v1 = point1 - vertex
    v2 = point2 - vertex

    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return float(angle)


def calculate_distance(
    point1: np.ndarray,
    point2: np.ndarray
) -> float:
    """
    Calculate Euclidean distance between two points.

    Args:
        point1: First point
        point2: Second point

    Returns:
        Distance
    """
    return float(np.linalg.norm(point1 - point2))


# Data Processing Functions
def normalize_keypoints(
    keypoints: np.ndarray,
    reference_point: Optional[np.ndarray] = None,
    scale: Optional[float] = None
) -> np.ndarray:
    """
    Normalize keypoints by translating to origin and scaling.

    Args:
        keypoints: Keypoints array (N, 2)
        reference_point: Reference point for translation (default: center)
        scale: Scale factor (default: auto from bounding box)

    Returns:
        Normalized keypoints
    """
    normalized = keypoints.copy()

    # Center at origin
    if reference_point is None:
        reference_point = np.mean(keypoints, axis=0)

    normalized = normalized - reference_point

    # Scale
    if scale is None:
        # Use max distance from center as scale
        distances = np.linalg.norm(normalized, axis=1)
        scale = np.max(distances)
        scale = scale if scale > 0 else 1.0

    normalized = normalized / scale

    return normalized


def create_sliding_windows(
    sequence: np.ndarray,
    window_size: int,
    stride: int = 1
) -> List[np.ndarray]:
    """
    Create sliding windows from a sequence.

    Args:
        sequence: Input sequence (T, features)
        window_size: Window size
        stride: Stride between windows

    Returns:
        List of windows
    """
    windows = []
    for i in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[i:i + window_size]
        windows.append(window)
    return windows


def pad_sequence(
    sequence: np.ndarray,
    target_length: int,
    pad_value: float = 0.0,
    padding: str = 'post'
) -> np.ndarray:
    """
    Pad sequence to target length.

    Args:
        sequence: Input sequence (T, features)
        target_length: Target length
        pad_value: Value to use for padding
        padding: 'pre' or 'post' padding

    Returns:
        Padded sequence
    """
    current_length = len(sequence)
    if current_length >= target_length:
        return sequence[:target_length]

    pad_length = target_length - current_length
    padding_shape = (pad_length,) + sequence.shape[1:]

    if padding == 'post':
        return np.vstack([sequence, np.full(padding_shape, pad_value)])
    else:
        return np.vstack([np.full(padding_shape, pad_value), sequence])


# Security Validation Functions
def validate_file_path(file_path: str) -> bool:
    """
    Validate file path to prevent directory traversal attacks.

    Args:
        file_path: Path to validate

    Returns:
        True if path is safe, False otherwise
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(file_path)
        current_dir = os.path.abspath(os.getcwd())

        # Check if path is within current directory or allowed subdirectories
        return abs_path.startswith(current_dir)
    except (AttributeError, TypeError):
        return False


def validate_file_size(file_path: str, max_size: int = MAX_VIDEO_SIZE) -> bool:
    """
    Validate file size to prevent DoS attacks.

    Args:
        file_path: Path to file
        max_size: Maximum allowed file size in bytes

    Returns:
        True if file size is acceptable, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            return False

        file_size = os.path.getsize(file_path)
        return file_size <= max_size
    except (OSError, TypeError):
        return False


def validate_file_extension(file_path: str, allowed_extensions: List[str]) -> bool:
    """
    Validate file extension.

    Args:
        file_path: Path to file
        allowed_extensions: List of allowed extensions

    Returns:
        True if extension is allowed, False otherwise
    """
    try:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in allowed_extensions
    except (AttributeError, TypeError):
        return False


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal and injection.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace potentially dangerous characters
    sanitized = re.sub(r'[^\w\-_\.]', '', filename)

    # Prevent hidden files
    if sanitized.startswith('.'):
        sanitized = 'file_' + sanitized

    # Limit length
    if len(sanitized) > 255:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:255-len(ext)] + ext

    return sanitized


# File System Functions
def ensure_dir(directory: str):
    """
    Ensure directory exists, create if it doesn't.

    Args:
        directory: Directory path
    """
    # Validate directory path before creating
    if validate_file_path(directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"Invalid directory path: {directory}")


def get_files_by_extension(
    directory: str,
    extensions: List[str],
    recursive: bool = True
) -> List[str]:
    """
    Get all files with specified extensions in directory.

    Args:
        directory: Directory to search
        extensions: List of file extensions (e.g., ['.mp4', '.avi'])
        recursive: Whether to search recursively

    Returns:
        List of file paths
    """
    files = []
    directory_path = Path(directory)

    if recursive:
        for ext in extensions:
            files.extend([str(p) for p in directory_path.rglob(f"*{ext}")])
    else:
        for ext in extensions:
            files.extend([str(p) for p in directory_path.glob(f"*{ext}")])

    return sorted(files)


def split_train_val_test(
    data_list: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data into train, validation, and test sets.

    Args:
        data_list: List of data file paths
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train, val, test) lists
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("Ratios must sum to 1.0")

    np.random.seed(random_seed)
    indices = np.random.permutation(len(data_list))

    train_end = int(len(data_list) * train_ratio)
    val_end = train_end + int(len(data_list) * val_ratio)

    train = [data_list[i] for i in indices[:train_end]]
    val = [data_list[i] for i in indices[train_end:val_end]]
    test = [data_list[i] for i in indices[val_end:]]

    logger = logging.getLogger(__name__)
    logger.info(f"Data split: Train={len(train)}, Val={len(val)}, Test={len(test)}")

    return train, val, test


# Metrics Functions
def calculate_count_metrics(
    predicted_counts: List[int],
    true_counts: List[int],
    tolerance: int = 1
) -> Dict[str, float]:
    """
    Calculate count-based evaluation metrics.

    Args:
        predicted_counts: List of predicted counts
        true_counts: List of true counts
        tolerance: Allowed absolute error for "correct" counts

    Returns:
        Dictionary of metrics
    """
    predicted = np.array(predicted_counts)
    true = np.array(true_counts)

    # Absolute errors
    abs_errors = np.abs(predicted - true)

    # MAE
    mae = np.mean(abs_errors)

    # MAPE (avoid division by zero)
    nonzero_mask = true > 0
    if np.any(nonzero_mask):
        mape = np.mean(abs_errors[nonzero_mask] / true[nonzero_mask]) * 100
    else:
        mape = 0.0

    # Count accuracy (within tolerance)
    correct = np.sum(abs_errors <= tolerance)
    count_accuracy = correct / len(true_counts) if len(true_counts) > 0 else 0.0

    # Exact accuracy
    exact_accuracy = np.sum(predicted == true) / len(true_counts) if len(true_counts) > 0 else 0.0

    return {
        'mae': float(mae),
        'mape': float(mape),
        'count_accuracy': float(count_accuracy),
        'exact_accuracy': float(exact_accuracy)
    }


# Timer Class
class Timer:
    """Simple timer for measuring execution time."""

    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None

    def start(self):
        """Start the timer."""
        self.start_time = datetime.now()
        return self

    def stop(self):
        """Stop the timer."""
        self.end_time = datetime.now()
        return self

    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.

        Returns:
            Elapsed time
        """
        if self.start_time is None:
            return 0.0

        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


# Progress Tracker
class ProgressTracker:
    """Simple progress tracker with ETA estimation."""

    def __init__(self, total: int, description: str = "Progress"):
        """
        Initialize progress tracker.

        Args:
            total: Total number of items
            description: Description string
        """
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = None

    def update(self, n: int = 1):
        """
        Update progress.

        Args:
            n: Number of items to add
        """
        if self.start_time is None:
            self.start_time = datetime.now()

        self.current = min(self.current + n, self.total)

        if self.current > 0:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            eta = elapsed / self.current * (self.total - self.current)
            self._print_progress(elapsed, eta)

    def _print_progress(self, elapsed: float, eta: float):
        """Print progress bar."""
        percent = self.current / self.total
        bar_length = 40
        filled = int(bar_length * percent)
        bar = '=' * filled + '-' * (bar_length - filled)

        print(f"\r{self.description}: [{bar}] {percent*100:.1f}% "
              f"({self.current}/{self.total}) | ETA: {eta:.1f}s", end='')

        if self.current == self.total:
            print()  # New line when complete

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.current < self.total:
            self.update(self.total - self.current)


# Color mapping for visualization
ACTION_COLORS = {
    'pushup': (0, 255, 255),      # Yellow
    'jumping_jack': (255, 0, 255), # Magenta
    'other': (128, 128, 128),     # Gray
    'none': (255, 255, 255)       # White
}


def get_action_color(action_type: str) -> Tuple[int, int, int]:
    """
    Get color for action type visualization.

    Args:
        action_type: Type of action

    Returns:
        RGB color tuple
    """
    return ACTION_COLORS.get(action_type.lower(), (255, 255, 255))
