"""
MediaPipe Pose Estimation Module

This module provides functionality for extracting human pose keypoints from video frames
using MediaPipe Pose Tasks API (for MediaPipe 0.10+), with normalization and angle calculation features.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Tuple, Optional, List, Dict
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model download URLs
MODEL_URLS = {
    'pose_landmarker_lite.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
    'pose_landmarker_full.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task',
    'pose_landmarker_heavy.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
}


class PoseEstimator:
    """
    MediaPipe Pose estimator for extracting human pose keypoints.

    Using MediaPipe's Tasks API for MediaPipe 0.10+ compatibility.
    """

    # MediaPipe Pose Landmarks (33 keypoints)
    LANDMARK_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
        'left_index', 'right_index', 'left_thumb', 'right_thumb',
        'left_hip', 'right_hip', 'left_knee', 'right_knee',
        'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    # Key landmark indices for angle calculations
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = True,  # Default to True for preprocessing compatibility
        model_complexity: int = 1,
        model_path: Optional[str] = None
    ):
        """
        Initialize Pose Estimator.

        Args:
            min_detection_confidence: Minimum confidence for pose detection (0.0-1.0)
            min_tracking_confidence: Minimum confidence for pose tracking (0.0-1.0)
            static_image_mode: Whether to treat input images as static
            model_complexity: Pose model complexity (0, 1, or 2)
            model_path: Optional path to model file (if None, uses default)
        """
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity

        # Initialize MediaPipe Pose Landmarker (Tasks API)
        try:
            # Map model complexity to the appropriate model
            model_asset_map = {
                0: 'pose_landmarker_lite.task',
                1: 'pose_landmarker_full.task',
                2: 'pose_landmarker_heavy.task'
            }

            # Use provided model path or default based on complexity
            if model_path:
                model_file = model_path
            else:
                model_file = model_asset_map.get(model_complexity, 'pose_landmarker_lite.task')

            # Check if model file exists
            import os
            if not os.path.exists(model_file):
                # Try to download it
                self._download_model(model_file)

            base_options = mp.tasks.BaseOptions(
                model_asset_path=model_file
            )

            self.options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE if static_image_mode
                           else mp.tasks.vision.RunningMode.VIDEO,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence,
                output_segmentation_masks=False
            )

            self.pose = mp.tasks.vision.PoseLandmarker.create_from_options(self.options)
            self.use_tasks_api = True

            logger.info(f"Pose Estimator initialized with Tasks API (complexity={model_complexity})")

        except Exception as e:
            # Fallback to legacy solutions API if Tasks API fails
            logger.warning(f"Tasks API not available, trying legacy API: {e}")
            self._init_legacy_api()
            self.use_tasks_api = False

    def _download_model(self, model_file: str):
        """Download MediaPipe model file if not exists."""
        if model_file in MODEL_URLS:
            import requests
            if not os.path.exists(model_file):
                logger.info(f"Downloading model {model_file}...")
                try:
                    response = requests.get(MODEL_URLS[model_file], stream=True)
                    response.raise_for_status()
                    with open(model_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"Model {model_file} downloaded successfully")
                except Exception as e:
                    logger.error(f"Failed to download {model_file}: {e}")
                    raise
        else:
            raise ValueError(f"Unknown model file: {model_file}")

    def _init_legacy_api(self):
        """Initialize legacy MediaPipe Pose API as fallback."""
        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles

            self.pose = self.mp_pose.Pose(
                static_image_mode=self.static_image_mode,
                model_complexity=self.model_complexity,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )

            logger.info(f"Pose Estimator initialized with legacy API (complexity={self.model_complexity})")

        except AttributeError as e:
            logger.error(f"Neither Tasks API nor legacy API available: {e}")
            raise RuntimeError("MediaPipe Pose is not available. Please install MediaPipe 0.9.x or 0.10.x")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Process a single video frame and extract pose keypoints.

        Args:
            frame: Input image frame (BGR format)
            timestamp_ms: Timestamp for video mode (required for non-static mode)

        Returns:
            Tuple of (keypoints, angles) where:
                keypoints: Normalized (x, y, z, visibility) array of shape (33, 4) or None
                angles: Dictionary of joint angles in degrees or None
        """
        if self.use_tasks_api:
            return self._process_frame_tasks(frame, timestamp_ms)
        else:
            return self._process_frame_legacy(frame)

    def _process_frame_tasks(
        self,
        frame: np.ndarray,
        timestamp_ms: Optional[int] = None
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Process frame using Tasks API."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Create MediaPipe image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # For video mode, we need a timestamp
            if timestamp_ms is None:
                timestamp_ms = int(np.datetime64('now').astype('int64') // 1000000)

            # Process frame
            if isinstance(self.pose, mp.tasks.vision.PoseLandmarker):
                result = self.pose.detect(mp_image)
            else:
                # Legacy API
                result = self.pose.process(frame_rgb)

            if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
                # Extract keypoints from first detected pose
                keypoints = self._extract_keypoints_tasks(result.pose_landmarks[0])

                # Calculate joint angles
                angles = self._calculate_angles(keypoints)

                return keypoints, angles

            return None, None

        except Exception as e:
            logger.error(f"Error processing frame with Tasks API: {e}")
            return None, None

    def _process_frame_legacy(
        self,
        frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Process frame using legacy API."""
        try:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame with MediaPipe Pose
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                # Extract keypoints
                keypoints = self._extract_keypoints_legacy(results.pose_landmarks)

                # Calculate joint angles
                angles = self._calculate_angles(keypoints)

                return keypoints, angles

            return None, None

        except Exception as e:
            logger.error(f"Error processing frame with legacy API: {e}")
            return None, None

    def _extract_keypoints_tasks(self, pose_landmark) -> np.ndarray:
        """
        Extract and normalize keypoints from MediaPipe Tasks API landmarks.

        Args:
            pose_landmark: MediaPipe PoseLandmarkerResult pose_landmark object

        Returns:
            Normalized keypoints array of shape (33, 4) with (x, y, z, visibility)
        """
        keypoints = np.zeros((33, 4), dtype=np.float32)

        if hasattr(pose_landmark, 'landmarks'):
            landmarks = pose_landmark.landmarks
            for i in range(min(33, len(landmarks))):
                lm = landmarks[i]
                keypoints[i, 0] = lm.x  # Normalized x coordinate (0-1)
                keypoints[i, 1] = lm.y  # Normalized y coordinate (0-1)
                keypoints[i, 2] = lm.z  # Normalized z coordinate
                keypoints[i, 3] = lm.visibility if hasattr(lm, 'visibility') else 1.0

        return keypoints

    def _extract_keypoints_legacy(self, landmarks) -> np.ndarray:
        """
        Extract and normalize keypoints from MediaPipe legacy landmarks.

        Args:
            landmarks: MediaPipe pose landmarks

        Returns:
            Normalized keypoints array of shape (33, 4) with (x, y, z, visibility)
        """
        keypoints = np.zeros((33, 4), dtype=np.float32)

        for i, landmark in enumerate(landmarks.landmark):
            keypoints[i, 0] = landmark.x  # Normalized x coordinate (0-1)
            keypoints[i, 1] = landmark.y  # Normalized y coordinate (0-1)
            keypoints[i, 2] = landmark.z  # Normalized z coordinate
            keypoints[i, 3] = landmark.visibility if hasattr(landmark, 'visibility') else 1.0

        return keypoints

    def _calculate_angles(self, keypoints: np.ndarray) -> Dict[str, float]:
        """
        Calculate joint angles from keypoints.

        Args:
            keypoints: Keypoints array of shape (33, 4)

        Returns:
            Dictionary of joint angles in degrees
        """
        angles = {}

        # Left arm angles
        angles['left_elbow'] = self._calculate_joint_angle(
            keypoints[self.LEFT_SHOULDER],
            keypoints[self.LEFT_ELBOW],
            keypoints[self.LEFT_WRIST]
        )
        angles['left_shoulder'] = self._calculate_joint_angle(
            keypoints[self.LEFT_ELBOW],
            keypoints[self.LEFT_SHOULDER],
            keypoints[self.LEFT_HIP]
        )

        # Right arm angles
        angles['right_elbow'] = self._calculate_joint_angle(
            keypoints[self.RIGHT_SHOULDER],
            keypoints[self.RIGHT_ELBOW],
            keypoints[self.RIGHT_WRIST]
        )
        angles['right_shoulder'] = self._calculate_joint_angle(
            keypoints[self.RIGHT_ELBOW],
            keypoints[self.RIGHT_SHOULDER],
            keypoints[self.RIGHT_HIP]
        )

        # Left leg angles
        angles['left_knee'] = self._calculate_joint_angle(
            keypoints[self.LEFT_HIP],
            keypoints[self.LEFT_KNEE],
            keypoints[self.LEFT_ANKLE]
        )
        angles['left_hip'] = self._calculate_joint_angle(
            keypoints[self.LEFT_SHOULDER],
            keypoints[self.LEFT_HIP],
            keypoints[self.LEFT_KNEE]
        )

        # Right leg angles
        angles['right_knee'] = self._calculate_joint_angle(
            keypoints[self.RIGHT_HIP],
            keypoints[self.RIGHT_KNEE],
            keypoints[self.RIGHT_ANKLE]
        )
        angles['right_hip'] = self._calculate_joint_angle(
            keypoints[self.RIGHT_SHOULDER],
            keypoints[self.RIGHT_HIP],
            keypoints[self.RIGHT_KNEE]
        )

        return angles

    def _calculate_joint_angle(
        self,
        point1: np.ndarray,
        vertex: np.ndarray,
        point2: np.ndarray
    ) -> float:
        """
        Calculate angle at vertex formed by point1-vertex-point2.

        Args:
            point1: First point coordinates
            vertex: Vertex point coordinates
            point2: Second point coordinates

        Returns:
            Angle in degrees (0-180)
        """
        # Vector 1: vertex -> point1
        v1 = point1[:2] - vertex[:2]
        # Vector 2: vertex -> point2
        v2 = point2[:2] - vertex[:2]

        # Calculate dot product and magnitudes
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        # Handle edge cases
        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Calculate angle
        cos_angle = np.clip(dot_product / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        angle_deg = np.degrees(angle_rad)

        return float(angle_deg)

    def process_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        visualize: bool = False
    ) -> List[Tuple[np.ndarray, Dict]]:
        """
        Process a video file and extract pose keypoints for all frames.

        Args:
            video_path: Path to input video file
            output_path: Optional path to save visualization video
            visualize: Whether to generate visualization

        Returns:
            List of (keypoints, angles) tuples for each frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        frames_data = []

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer for visualization
        writer = None
        if visualize and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        timestamp_ms = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            keypoints, angles = self.process_frame(frame, timestamp_ms)

            if keypoints is not None:
                frames_data.append((keypoints, angles))

                # Draw visualization
                if visualize and writer is not None:
                    frame_vis = self.visualize_pose(frame, keypoints, angles)
                    writer.write(frame_vis)

            frame_count += 1
            timestamp_ms += int(1000 / fps) if fps > 0 else 33

            if frame_count % 30 == 0:
                logger.info(f"Processed {frame_count} frames")

        cap.release()
        if writer is not None:
            writer.release()

        logger.info(f"Processed {frame_count} frames, extracted {len(frames_data)} pose estimates")
        return frames_data

    def visualize_pose(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        angles: Optional[Dict] = None,
        thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize pose keypoints and connections on frame.

        Args:
            frame: Input image frame
            keypoints: Keypoints array of shape (33, 4)
            angles: Optional dictionary of joint angles to display
            thickness: Line thickness for drawing

        Returns:
            Frame with pose visualization
        """
        height, width = frame.shape[:2]
        vis_frame = frame.copy()

        # Define pose connections (pairs of landmark indices)
        connections = [
            (11, 12),  # Shoulder
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (11, 23), (23, 25), (25, 27),  # Left leg
            (12, 24), (24, 26), (26, 28),  # Right leg
            (23, 24),  # Hip
            (25, 27), (26, 28),  # Legs
        ]

        # Draw connections
        for connection in connections:
            start_idx, end_idx = connection
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]

            # Only draw if both points are visible
            if start_point[3] > 0.5 and end_point[3] > 0.5:
                start_pos = (int(start_point[0] * width), int(start_point[1] * height))
                end_pos = (int(end_point[0] * width), int(end_point[1] * height))
                cv2.line(vis_frame, start_pos, end_pos, (0, 255, 0), thickness)

        # Draw keypoints
        for i, point in enumerate(keypoints):
            if point[3] > 0.5:  # Only draw visible points
                x, y = int(point[0] * width), int(point[1] * height)
                color = (0, 0, 255) if i in [15, 16, 27, 28] else (255, 0, 0)
                cv2.circle(vis_frame, (x, y), 3, color, -1)

        # Draw angles if provided
        if angles:
            y_offset = 30
            for joint_name, angle in angles.items():
                if 'elbow' in joint_name or 'knee' in joint_name:
                    cv2.putText(
                        vis_frame,
                        f"{joint_name}: {angle:.1f}°",
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1
                    )
                    y_offset += 20

        return vis_frame

    def normalize_keypoints(
        self,
        keypoints: np.ndarray,
        method: str = "bounding_box"
    ) -> np.ndarray:
        """
        Normalize keypoints to reduce position and scale variance.

        Args:
            keypoints: Keypoints array of shape (33, 4)
            method: Normalization method ('bounding_box', 'hip_center', 'shoulder_width')

        Returns:
            Normalized keypoints array
        """
        normalized = keypoints.copy()

        if method == "bounding_box":
            # Normalize to bounding box
            visible_points = keypoints[keypoints[:, 3] > 0.5]
            if len(visible_points) > 0:
                min_x, min_y = visible_points[:, 0].min(), visible_points[:, 1].min()
                max_x, max_y = visible_points[:, 0].max(), visible_points[:, 1].max()

                width = max_x - min_x
                height = max_y - min_y
                scale = max(width, height)

                if scale > 0:
                    normalized[:, 0] = (keypoints[:, 0] - min_x) / scale
                    normalized[:, 1] = (keypoints[:, 1] - min_y) / scale

        elif method == "hip_center":
            # Normalize to hip center
            hip_center = (
                (keypoints[self.LEFT_HIP, :2] + keypoints[self.RIGHT_HIP, :2]) / 2
            )
            shoulder_width = np.linalg.norm(
                keypoints[self.LEFT_SHOULDER, :2] - keypoints[self.RIGHT_SHOULDER, :2]
            )

            if shoulder_width > 0:
                normalized[:, 0] = (keypoints[:, 0] - hip_center[0]) / shoulder_width
                normalized[:, 1] = (keypoints[:, 1] - hip_center[1]) / shoulder_width

        return normalized

    def extract_features(
        self,
        keypoints: np.ndarray,
        angles: Dict[str, float]
    ) -> np.ndarray:
        """
        Extract feature vector from keypoints and angles for ML model input.

        Args:
            keypoints: Keypoints array of shape (33, 4)
            angles: Dictionary of joint angles

        Returns:
            Feature vector of shape (50,) combining normalized keypoints and angles
        """
        # Normalize keypoints
        norm_keypoints = self.normalize_keypoints(keypoints, method="hip_center")

        # Flatten x, y coordinates (excluding z and visibility)
        xy_coords = norm_keypoints[:, :2].flatten()  # 66 features

        # Extract angles (8 angles)
        angle_values = np.array([
            angles.get('left_elbow', 0),
            angles.get('right_elbow', 0),
            angles.get('left_shoulder', 0),
            angles.get('right_shoulder', 0),
            angles.get('left_knee', 0),
            angles.get('right_knee', 0),
            angles.get('left_hip', 0),
            angles.get('right_hip', 0)
        ])

        # Normalize angles to 0-1 range
        norm_angles = angle_values / 180.0

        # Combine features
        features = np.concatenate([xy_coords, norm_angles])  # 74 features

        # Select most important features (optional)
        important_indices = [
            # Torso points
            self.LEFT_SHOULDER * 2, self.LEFT_SHOULDER * 2 + 1,
            self.RIGHT_SHOULDER * 2, self.RIGHT_SHOULDER * 2 + 1,
            self.LEFT_HIP * 2, self.LEFT_HIP * 2 + 1,
            self.RIGHT_HIP * 2, self.RIGHT_HIP * 2 + 1,
            # Arms
            self.LEFT_ELBOW * 2, self.LEFT_ELBOW * 2 + 1,
            self.RIGHT_ELBOW * 2, self.RIGHT_ELBOW * 2 + 1,
            self.LEFT_WRIST * 2, self.LEFT_WRIST * 2 + 1,
            self.RIGHT_WRIST * 2, self.RIGHT_WRIST * 2 + 1,
            # Legs
            self.LEFT_KNEE * 2, self.LEFT_KNEE * 2 + 1,
            self.RIGHT_KNEE * 2, self.RIGHT_KNEE * 2 + 1,
            self.LEFT_ANKLE * 2, self.LEFT_ANKLE * 2 + 1,
            self.RIGHT_ANKLE * 2, self.RIGHT_ANKLE * 2 + 1
        ]

        selected_features = features[important_indices]
        final_features = np.concatenate([selected_features, norm_angles])

        return final_features  # 42 features

    def close(self):
        """Release resources."""
        if hasattr(self, 'pose') and hasattr(self.pose, 'close'):
            try:
                self.pose.close()
            except:
                pass

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()


class BatchPoseEstimator:
    """
    Batch processor for pose estimation on multiple videos.

    This class handles processing multiple video files efficiently with
    progress tracking and error handling.
    """

    def __init__(self, **pose_estimator_kwargs):
        """Initialize batch processor with pose estimator arguments."""
        self.pose_estimator = PoseEstimator(**pose_estimator_kwargs)
        logger.info("Batch Pose Estimator initialized")

    def process_batch(
        self,
        video_paths: List[str],
        output_dir: Optional[str] = None,
        visualize: bool = False
    ) -> Dict[str, List[Tuple[np.ndarray, Dict]]]:
        """
        Process multiple videos and extract pose keypoints.

        Args:
            video_paths: List of video file paths to process
            output_dir: Optional directory to save visualization videos
            visualize: Whether to generate visualization videos

        Returns:
            Dictionary mapping video paths to their frame data
        """
        results = {}

        for video_path in video_paths:
            logger.info(f"Processing video: {video_path}")

            output_path = None
            if output_dir and visualize:
                import os
                os.makedirs(output_dir, exist_ok=True)
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_dir, f"{video_name}_vis.mp4")

            try:
                frames_data = self.pose_estimator.process_video(
                    video_path,
                    output_path=output_path,
                    visualize=visualize
                )
                results[video_path] = frames_data
                logger.info(f"Successfully processed {len(frames_data)} frames")
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")
                results[video_path] = []

        return results

    def close(self):
        """Release resources."""
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.close()

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()
