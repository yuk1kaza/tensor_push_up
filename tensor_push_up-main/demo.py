"""
Unified demo entrypoints for pose display and exercise counting.

The counting demos now use the same ActionInference pipeline as infer.py so
camera and offline video behavior stay consistent.
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.infer import ActionInference
from src.pose_estimator import PoseEstimator
from src.utils import VideoReader, draw_counter_display


def _smooth_pose(
    keypoints: np.ndarray,
    angles: dict,
    previous_keypoints: Optional[np.ndarray],
    previous_angles: Optional[dict],
    alpha: float = 0.35,
):
    """EMA smoothing for pose-only webcam overlays."""
    if previous_keypoints is None:
        smoothed_keypoints = keypoints.copy()
    else:
        smoothed_keypoints = alpha * keypoints + (1.0 - alpha) * previous_keypoints

    if previous_angles is None:
        smoothed_angles = dict(angles)
    else:
        smoothed_angles = {}
        for joint_name, angle in angles.items():
            prev = previous_angles.get(joint_name, angle)
            smoothed_angles[joint_name] = alpha * angle + (1.0 - alpha) * prev

    return smoothed_keypoints, smoothed_angles


def _configure_capture(cap: cv2.VideoCapture):
    """Reduce webcam latency for interactive demos."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def _resolve_default_model_path() -> Optional[str]:
    candidates = [
        Path("models/checkpoints/best_model.keras"),
        Path("models/exported/action_classifier.h5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _detect_pose_demo_action(keypoints: np.ndarray, angles: dict) -> str:
    """Cheap heuristic action hint for pose-only demo mode."""
    if keypoints is None or angles is None:
        return "None"

    left_elbow = angles.get("left_elbow", 180.0)
    right_elbow = angles.get("right_elbow", 180.0)
    avg_elbow = (left_elbow + right_elbow) / 2.0

    left_wrist = keypoints[15, :2]
    right_wrist = keypoints[16, :2]
    left_shoulder = keypoints[11, :2]
    right_shoulder = keypoints[12, :2]
    left_ankle = keypoints[27, :2]
    right_ankle = keypoints[28, :2]

    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    ankle_distance = np.linalg.norm(left_ankle - right_ankle)
    wrists_up = left_wrist[1] < left_shoulder[1] and right_wrist[1] < right_shoulder[1]

    if shoulder_width > 0 and ankle_distance / shoulder_width > 1.2 and wrists_up:
        return "Jumping Jack"
    if avg_elbow < 150:
        return "Push-up"
    return "Other"


def _frame_stream(video_path: Optional[str], camera_idx: int) -> tuple[Iterable[np.ndarray], Optional[cv2.VideoCapture]]:
    """Return a frame iterator plus capture handle for webcam cleanup."""
    if video_path:
        return VideoReader(video_path), None

    cap = cv2.VideoCapture(camera_idx)
    _configure_capture(cap)

    def webcam_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame

    return webcam_frames(), cap


def demo_pose_estimation(video_path: str = None, camera_idx: int = 0):
    """Demo pose estimation functionality with a heuristic action hint."""
    print("=" * 60)
    print("Pose Estimation Demo")
    print("=" * 60)

    estimator = PoseEstimator(
        static_image_mode=False,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_idx = 0
    last_time = time.perf_counter()
    frames, cap = _frame_stream(video_path, camera_idx)
    smoothed_keypoints = None
    smoothed_angles = None

    try:
        for frame in frames:
            keypoints, angles = estimator.process_frame(frame, timestamp_ms=frame_idx * 33)

            vis_frame = frame
            current_time = time.perf_counter()
            fps = 1.0 / max(current_time - last_time, 1e-6)
            last_time = current_time

            if keypoints is not None:
                if cap is not None:
                    keypoints, angles = _smooth_pose(
                        keypoints,
                        angles,
                        smoothed_keypoints,
                        smoothed_angles,
                    )
                    smoothed_keypoints = keypoints
                    smoothed_angles = angles

                vis_frame = estimator.visualize_pose(frame, keypoints, angles)
                action_type = _detect_pose_demo_action(keypoints, angles)
                vis_frame = draw_counter_display(
                    vis_frame,
                    pushup_count=0,
                    jumping_jack_count=0,
                    action_type=action_type,
                    confidence=0.0,
                    fps=fps,
                )

            cv2.imshow("Pose Estimation Demo", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        estimator.close()

    print("Pose estimation demo complete!")


def _run_count_demo(
    title: str,
    exercise_type: str,
    video_path: Optional[str] = None,
    camera_idx: int = 0,
):
    """Run counting demo through the same pipeline as infer.py."""
    print("=" * 60)
    print(title)
    print("=" * 60)

    model_path = _resolve_default_model_path()
    inference = ActionInference(
        model_path=model_path,
        exercise_type=exercise_type,
        use_model=bool(model_path),
        realtime_smoothing=video_path is None,
    )

    frames, cap = _frame_stream(video_path, camera_idx)

    try:
        for frame in frames:
            annotated_frame, _ = inference.process_frame(frame)
            cv2.imshow(title, annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        inference.close()

    if exercise_type == "pushup":
        final_count = inference.counter.count if inference.counter else 0
        print(f"Final Push-up Count: {final_count}")
    else:
        final_count = inference.counter.count if inference.counter else 0
        print(f"Final Jumping Jack Count: {final_count}")


def demo_pushup_counter(video_path: str = None, camera_idx: int = 0):
    _run_count_demo("Push-up Counter Demo", "pushup", video_path, camera_idx)


def demo_jumping_jack_counter(video_path: str = None, camera_idx: int = 0):
    _run_count_demo("Jumping Jack Counter Demo", "jumping_jack", video_path, camera_idx)


def main():
    """Main entry point for demo."""
    parser = argparse.ArgumentParser(description="Demo script for Tensor Push Up")
    parser.add_argument("--mode", type=str, choices=["pose", "pushup", "jumping_jack"], default="pose", help="Demo mode to run")
    parser.add_argument("--source", type=str, help="Video file path (default: webcam)")
    parser.add_argument("--camera", type=int, default=0, help="Webcam camera index")

    args = parser.parse_args()

    if args.mode == "pose":
        demo_pose_estimation(args.source, args.camera)
    elif args.mode == "pushup":
        demo_pushup_counter(args.source, args.camera)
    elif args.mode == "jumping_jack":
        demo_jumping_jack_counter(args.source, args.camera)


if __name__ == "__main__":
    main()
