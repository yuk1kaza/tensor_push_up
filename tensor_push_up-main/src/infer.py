"""
Inference Script Module

This module provides real-time and offline inference capabilities for the
action classification model, including pose estimation, action prediction,
and action counting using state machines.
"""

import os
import sys
import argparse
import time
import logging
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pose_estimator import PoseEstimator
from src.counter import ExerciseCounter, PushUpCounter, JumpingJackCounter
from src.model import ModelInference, get_action_class_id, get_action_class_name
from src.utils import (
    setup_logging, VideoReader, VideoWriter,
    draw_pose_on_image, draw_text_overlay, draw_counter_display,
    get_action_color, Timer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionInference:
    """
    Main inference class for real-time action recognition and counting.

    Handles:
    1. Pose estimation from video frames
    2. Action classification using trained model
    3. Action counting using state machines
    4. Visualization of results
    5. Real-time and offline inference modes
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        exercise_type: str = "auto",
        use_model: bool = True,
        window_size: int = 30,
        confidence_threshold: float = 0.6,
        smoothing_window: int = 5,
        display_skeleton: bool = True,
        display_counter: bool = True
    ):
        """
        Initialize the inference system.

        Args:
            model_path: Path to trained model checkpoint
            exercise_type: Type of exercise ('pushup', 'jumping_jack', 'auto', or 'all')
            use_model: Whether to use model for action classification
            window_size: Window size for temporal features
            confidence_threshold: Minimum confidence for predictions
            smoothing_window: Window size for prediction smoothing
            display_skeleton: Whether to display pose skeleton
            display_counter: Whether to display counter overlay
        """
        self.exercise_type = exercise_type
        self.use_model = use_model
        self.display_skeleton = display_skeleton
        self.display_counter = display_counter
        self.confidence_threshold = confidence_threshold

        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize model if specified
        self.model_inference = None
        if use_model and model_path and os.path.exists(model_path):
            self.model_inference = ModelInference(
                model_path=model_path,
                window_size=window_size,
                smoothing_window=smoothing_window,
                confidence_threshold=confidence_threshold
            )
            logger.info(f"Model loaded from {model_path}")
        elif use_model:
            logger.warning("Model path not specified or not found, using rule-based only")

        # Initialize counter
        if exercise_type == "auto" or exercise_type == "all":
            self.counter = ExerciseCounter()
        elif exercise_type == "pushup":
            self.counter = PushUpCounter()
        elif exercise_type == "jumping_jack":
            self.counter = JumpingJackCounter()
        else:
            self.counter = None

        # Feature buffer for temporal processing
        self.feature_buffer = []
        self.max_buffer_size = window_size

        # Current state
        self.current_action = "None"
        self.current_confidence = 0.0
        self.frame_count = 0

        # FPS tracking
        self.last_time = time.time()
        self.fps = 0.0

        # MediaPipe connections for visualization
        self.mp_connections = self.pose_estimator.mp_pose.POSE_CONNECTIONS

        logger.info(f"ActionInference initialized with exercise_type={exercise_type}")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process a single frame for action detection and counting.

        Args:
            frame: Input video frame

        Returns:
            Tuple of (annotated_frame, results_dict)
        """
        self.frame_count += 1

        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_time
        if dt > 0:
            self.fps = 1.0 / dt
        self.last_time = current_time

        # Extract pose
        keypoints, angles = self.pose_estimator.process_frame(frame)

        # Initialize results
        results = {
            'keypoints': keypoints,
            'angles': angles,
            'action': 'none',
            'confidence': 0.0,
            'pushup_count': 0,
            'jumping_jack_count': 0,
            'state': 'neutral',
            'fps': self.fps
        }

        annotated_frame = frame.copy()

        if keypoints is None:
            # No pose detected
            return annotated_frame, results

        # Extract features
        features = self.pose_estimator.extract_features(keypoints, angles)

        # Add to feature buffer
        self.feature_buffer.append(features)
        if len(self.feature_buffer) > self.max_buffer_size:
            self.feature_buffer.pop(0)

        # Predict action using model
        action = "none"
        confidence = 0.0

        if self.use_model and self.model_inference is not None and len(self.feature_buffer) == self.max_buffer_size:
            feature_array = np.array(self.feature_buffer)
            action, confidence = self.model_inference.predict(feature_array)
        else:
            # Rule-based action detection
            action = self._detect_action_rule_based(angles)

        self.current_action = action
        self.current_confidence = confidence

        # Update counters
        if self.counter is not None:
            if isinstance(self.counter, ExerciseCounter):
                counter_results = self.counter.process_frame(
                    keypoints, angles,
                    exercise_type=self.exercise_type if self.exercise_type != 'auto' else None,
                    frame_idx=self.frame_count
                )
                results['pushup_count'] = counter_results['pushup']['count']
                results['jumping_jack_count'] = counter_results['jumping_jack']['count']
                results['state'] = counter_results['pushup']['state'] if action == 'pushup' else counter_results['jumping_jack']['state']
            else:
                counter_results = self.counter.process_frame(keypoints, angles, frame_idx=self.frame_count)
                if isinstance(self.counter, PushUpCounter):
                    results['pushup_count'] = counter_results['count']
                    results['state'] = counter_results['state']
                else:
                    results['jumping_jack_count'] = counter_results['count']
                    results['state'] = counter_results['state']

        # Update results
        results['action'] = action
        results['confidence'] = confidence

        # Annotate frame
        annotated_frame = self._annotate_frame(
            frame, keypoints, action, confidence,
            results['pushup_count'], results['jumping_jack_count'],
            results['state'], self.fps
        )

        return annotated_frame, results

    def _detect_action_rule_based(self, angles: Dict[str, float]) -> str:
        """
        Detect action type using rule-based logic.

        Args:
            angles: Dictionary of joint angles

        Returns:
            Detected action type
        """
        left_elbow = angles.get('left_elbow', 180)
        right_elbow = angles.get('right_elbow', 180)
        avg_elbow = (left_elbow + right_elbow) / 2

        left_knee = angles.get('left_knee', 180)
        right_knee = angles.get('right_knee', 180)
        avg_knee = (left_knee + right_knee) / 2

        # Simple rule-based detection
        # Push-up: elbow angle changes significantly, knees relatively straight
        if avg_elbow < 150:
            return 'pushup'
        # Jumping jack: legs are moving
        elif avg_knee < 140:
            return 'jumping_jack'
        else:
            return 'other'

    def _annotate_frame(
        self,
        frame: np.ndarray,
        keypoints: np.ndarray,
        action: str,
        confidence: float,
        pushup_count: int,
        jumping_jack_count: int,
        state: str,
        fps: float
    ) -> np.ndarray:
        """
        Annotate frame with pose skeleton and counter display.

        Args:
            frame: Input frame
            keypoints: Pose keypoints
            action: Detected action
            confidence: Detection confidence
            pushup_count: Push-up count
            jumping_jack_count: Jumping jack count
            state: Current counter state
            fps: Current FPS

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        # Draw pose skeleton
        if self.display_skeleton:
            annotated = draw_pose_on_image(
                annotated,
                keypoints,
                list(self.mp_connections),
                keypoint_color=(0, 255, 0),
                connection_color=(0, 255, 0),
                thickness=2
            )

        # Draw counter display
        if self.display_counter:
            annotated = draw_counter_display(
                annotated,
                pushup_count=pushup_count,
                jumping_jack_count=jumping_jack_count,
                action_type=action.title(),
                confidence=confidence,
                fps=fps
            )

        # Draw state indicator
        state_color = get_action_color(action)
        cv2.circle(annotated, (30, annotated.shape[0] - 30), 20, state_color, -1)
        cv2.putText(
            annotated,
            state.upper(),
            (60, annotated.shape[0] - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            state_color,
            2
        )

        return annotated

    def reset(self):
        """Reset counters and buffers."""
        if self.counter:
            self.counter.reset()
        self.feature_buffer.clear()
        self.frame_count = 0
        logger.info("Inference system reset")

    def close(self):
        """Release resources."""
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.close()
        logger.info("Inference system closed")


def run_webcam_inference(
    model_path: Optional[str] = None,
    exercise_type: str = "auto",
    use_model: bool = True,
    camera_index: int = 0,
    save_video: bool = False,
    output_path: Optional[str] = None
):
    """
    Run real-time inference from webcam.

    Args:
        model_path: Path to trained model
        exercise_type: Type of exercise to detect
        use_model: Whether to use model for classification
        camera_index: Webcam camera index
        save_video: Whether to save output video
        output_path: Path to save output video
    """
    # Initialize inference
    inference = ActionInference(
        model_path=model_path,
        exercise_type=exercise_type,
        use_model=use_model
    )

    # Initialize video capture
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Failed to open camera {camera_index}")
        return

    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    logger.info(f"Camera opened: {width}x{height} @ {fps}fps")

    # Initialize video writer if saving
    writer = None
    if save_video:
        if output_path is None:
            output_path = f"webcam_output_{int(time.time())}.mp4"
        writer = VideoWriter(output_path, fps=30, resolution=(width, height))

    logger.info("Starting webcam inference. Press 'q' to quit, 'r' to reset counter.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated_frame, results = inference.process_frame(frame)

            # Write to video
            if writer is not None:
                writer.write_frame(annotated_frame)

            # Display
            cv2.imshow("Tensor Push Up - Real-time Inference", annotated_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                inference.reset()
                logger.info("Counter reset")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")

    finally:
        cap.release()
        if writer is not None:
            writer.close()
        cv2.destroyAllWindows()
        inference.close()

    logger.info(f"Final counts - Push-ups: {results['pushup_count']}, "
               f"Jumping Jacks: {results['jumping_jack_count']}")


def run_video_inference(
    video_path: str,
    model_path: Optional[str] = None,
    exercise_type: str = "auto",
    use_model: bool = True,
    save_video: bool = True,
    output_path: Optional[str] = None
) -> Dict:
    """
    Run inference on a video file.

    Args:
        video_path: Path to input video
        model_path: Path to trained model
        exercise_type: Type of exercise to detect
        use_model: Whether to use model for classification
        save_video: Whether to save annotated output video
        output_path: Path to save output video

    Returns:
        Dictionary with inference results
    """
    # Initialize inference
    inference = ActionInference(
        model_path=model_path,
        exercise_type=exercise_type,
        use_model=use_model
    )

    # Open video
    reader = VideoReader(video_path)

    # Determine output path
    if save_video:
        if output_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = f"{video_name}_output.mp4"
    else:
        output_path = None

    # Initialize video writer
    writer = None
    if output_path:
        writer = VideoWriter(output_path, reader.fps, (reader.width, reader.height))

    logger.info(f"Processing video: {video_path}")
    logger.info(f"Total frames: {reader.frame_count}")

    # Process frames
    all_results = []
    pushup_count = 0
    jumping_jack_count = 0

    with ProgressTracker(reader.frame_count, "Processing frames") as pbar:
        for frame_idx, frame in enumerate(reader):
            annotated_frame, results = inference.process_frame(frame)

            if writer is not None:
                writer.write_frame(annotated_frame)

            all_results.append(results)
            pushup_count = results['pushup_count']
            jumping_jack_count = results['jumping_jack_count']

            pbar.update(1)

    # Close resources
    reader.close()
    if writer is not None:
        writer.close()
    inference.close()

    # Compile final results
    final_results = {
        'video_path': video_path,
        'output_path': output_path,
        'total_frames': reader.frame_count,
        'duration': reader.duration,
        'final_pushup_count': pushup_count,
        'final_jumping_jack_count': jumping_jack_count,
        'all_results': all_results
    }

    logger.info(f"Video processing complete!")
    logger.info(f"  Final counts - Push-ups: {pushup_count}, Jumping Jacks: {jumping_jack_count}")

    return final_results


def run_batch_inference(
    video_dir: str,
    model_path: Optional[str] = None,
    exercise_type: str = "auto",
    use_model: bool = True,
    output_dir: str = "results/inference",
    save_videos: bool = True
) -> Dict:
    """
    Run inference on multiple videos in a directory.

    Args:
        video_dir: Directory containing videos to process
        model_path: Path to trained model
        exercise_type: Type of exercise to detect
        use_model: Whether to use model for classification
        output_dir: Directory to save results
        save_videos: Whether to save annotated output videos

    Returns:
        Dictionary with all results
    """
    from src.utils import get_files_by_extension, ensure_dir

    ensure_dir(output_dir)

    # Get all video files
    video_files = get_files_by_extension(
        video_dir,
        ['.mp4', '.avi', '.mov', '.mkv'],
        recursive=True
    )

    logger.info(f"Found {len(video_files)} videos to process")

    all_results = {}

    for video_path in video_files:
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_dir, f"{os.path.splitext(video_name)[0]}_output.mp4")

        logger.info(f"Processing: {video_name}")

        results = run_video_inference(
            video_path=video_path,
            model_path=model_path,
            exercise_type=exercise_type,
            use_model=use_model,
            save_video=save_videos,
            output_path=output_path
        )

        all_results[video_name] = {
            'pushup_count': results['final_pushup_count'],
            'jumping_jack_count': results['final_jumping_jack_count'],
            'output_path': results['output_path']
        }

    # Save summary
    import json
    with open(os.path.join(output_dir, 'batch_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Batch processing complete! Results saved to {output_dir}")

    return all_results


def main():
    """Main entry point for inference."""
    parser = argparse.ArgumentParser(
        description="Real-time and offline inference for action recognition and counting"
    )

    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=str,
                            help="Input source: camera index (e.g., 0) or video path")
    parser.add_argument("--batch-dir", type=str,
                       help="Process all videos in directory (alternative to --source)")

    # Model settings
    parser.add_argument("--model", type=str,
                       help="Path to trained model checkpoint")
    parser.add_argument("--no-model", action="store_true",
                       help="Use rule-based detection only (no model)")

    # Exercise type
    parser.add_argument("--exercise", type=str,
                       choices=["pushup", "jumping_jack", "auto", "all"],
                       default="auto",
                       help="Type of exercise to detect/count")

    # Output settings
    parser.add_argument("--output", type=str,
                       help="Output video path (for single video)")
    parser.add_argument("--output-dir", type=str, default="results/inference",
                       help="Output directory for batch processing")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output video")

    # Display settings
    parser.add_argument("--hide-skeleton", action="store_true",
                       help="Don't display pose skeleton")
    parser.add_argument("--hide-counter", action="store_true",
                       help="Don't display counter overlay")

    args = parser.parse_args()

    # Determine if using model
    use_model = not args.no_model

    # Check if source is camera or video
    try:
        source_value = int(args.source)
        is_camera = True
    except (ValueError, TypeError):
        source_value = args.source
        is_camera = False

    # Run inference
    if args.batch_dir:
        # Batch processing
        run_batch_inference(
            video_dir=args.batch_dir,
            model_path=args.model,
            exercise_type=args.exercise,
            use_model=use_model,
            output_dir=args.output_dir,
            save_videos=not args.no_save
        )
    elif is_camera:
        # Webcam inference
        run_webcam_inference(
            model_path=args.model,
            exercise_type=args.exercise,
            use_model=use_model,
            camera_index=source_value,
            save_video=False
        )
    else:
        # Video file inference
        run_video_inference(
            video_path=source_value,
            model_path=args.model,
            exercise_type=args.exercise,
            use_model=use_model,
            save_video=not args.no_save,
            output_path=args.output
        )


if __name__ == "__main__":
    main()
