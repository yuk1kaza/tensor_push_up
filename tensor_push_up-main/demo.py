"""
Demo Script for Tensor Push Up Project

This script provides a quick demonstration of the key functionality:
1. Pose estimation from a video or webcam
2. Action detection with state machine counting
3. Visualization of results
"""

import argparse
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.pose_estimator import PoseEstimator
from src.counter import PushUpCounter, JumpingJackCounter
from src.utils import VideoReader, draw_counter_display, Timer
import cv2
import numpy as np


def demo_pose_estimation(video_path: str = None, camera_idx: int = 0):
    """
    Demo pose estimation functionality.

    Args:
        video_path: Path to video file (None for webcam)
        camera_idx: Webcam camera index
    """
    print("=" * 60)
    print("Pose Estimation Demo")
    print("=" * 60)

    estimator = PoseEstimator()

    if video_path:
        # Process video file
        print(f"Processing video: {video_path}")

        with VideoReader(video_path) as reader:
            for frame in reader:
                keypoints, angles = estimator.process_frame(frame)

                if keypoints is not None:
                    # Visualize pose
                    vis_frame = estimator.visualize_pose(frame, keypoints, angles)

                    # Add text overlay
                    text_lines = [
                        ("Pose Estimation Demo", (0, 255, 255)),
                        (f"Elbow Angle: {angles['left_elbow']:.1f}°", (0, 255, 0)),
                        ("Press 'q' to quit", (255, 255, 255))
                    ]

                    vis_frame = draw_counter_display(
                        vis_frame,
                        pushup_count=0,
                        jumping_jack_count=0,
                        action_type="Testing",
                        confidence=0.0,
                        fps=30.0
                    )

                    cv2.imshow("Pose Estimation Demo", vis_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    else:
        # Process webcam
        print(f"Opening webcam: {camera_idx}")
        cap = cv2.VideoCapture(camera_idx)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, angles = estimator.process_frame(frame)

            if keypoints is not None:
                vis_frame = estimator.visualize_pose(frame, keypoints, angles)

                vis_frame = draw_counter_display(
                    vis_frame,
                    pushup_count=0,
                    jumping_jack_count=0,
                    action_type="Testing",
                    confidence=0.0,
                    fps=30.0
                )

                cv2.imshow("Pose Estimation Demo", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

    cv2.destroyAllWindows()
    estimator.close()

    print("Pose estimation demo complete!")


def demo_pushup_counter(video_path: str = None, camera_idx: int = 0):
    """
    Demo push-up counter.

    Args:
        video_path: Path to video file (None for webcam)
        camera_idx: Webcam camera index
    """
    print("=" * 60)
    print("Push-up Counter Demo")
    print("=" * 60)

    estimator = PoseEstimator()
    counter = PushUpCounter(
        high_angle_threshold=150.0,
        low_angle_threshold=90.0,
        stability_frames=3,
        cooldown_frames=10
    )

    frame_idx = 0

    if video_path:
        with VideoReader(video_path) as reader:
            for frame in reader:
                keypoints, angles = estimator.process_frame(frame)
                result = counter.process_frame(keypoints, angles, frame_idx)

                vis_frame = draw_counter_display(
                    frame,
                    pushup_count=result['count'],
                    jumping_jack_count=0,
                    action_type="Push-up",
                    confidence=0.0,
                    fps=30.0
                )

                if result['transition']:
                    print(f"Count! Total: {result['count']}")

                cv2.imshow("Push-up Counter Demo", vis_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_idx += 1
    else:
        cap = cv2.VideoCapture(camera_idx)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, angles = estimator.process_frame(frame)
            result = counter.process_frame(keypoints, angles, frame_idx)

            vis_frame = draw_counter_display(
                frame,
                pushup_count=result['count'],
                jumping_jack_count=0,
                action_type="Push-up",
                confidence=0.0,
                fps=30.0
            )

            cv2.imshow("Push-up Counter Demo", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        cap.release()

    cv2.destroyAllWindows()
    estimator.close()

    print(f"Final Push-up Count: {counter.count}")


def demo_jumping_jack_counter(video_path: str = None, camera_idx: int = 0):
    """
    Demo jumping jack counter.

    Args:
        video_path: Path to video file (None for webcam)
        camera_idx: Webcam camera index
    """
    print("=" * 60)
    print("Jumping Jack Counter Demo")
    print("=" * 60)

    estimator = PoseEstimator()
    counter = JumpingJackCounter(
        open_ankle_threshold=0.3,
        closed_ankle_threshold=0.1,
        stability_frames=3,
        cooldown_frames=10
    )

    frame_idx = 0

    if video_path:
        with VideoReader(video_path) as reader:
            for frame in reader:
                keypoints, angles = estimator.process_frame(frame)
                result = counter.process_frame(keypoints, angles, frame_idx)

                vis_frame = draw_counter_display(
                    frame,
                    pushup_count=0,
                    jumping_jack_count=result['count'],
                    action_type="Jumping Jack",
                    confidence=0.0,
                    fps=30.0
                )

                cv2.imshow("Jumping Jack Counter Demo", vis_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_idx += 1
    else:
        cap = cv2.VideoCapture(camera_idx)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, angles = estimator.process_frame(frame)
            result = counter.process_frame(keypoints, angles, frame_idx)

            vis_frame = draw_counter_display(
                frame,
                pushup_count=0,
                jumping_jack_count=result['count'],
                action_type="Jumping Jack",
                confidence=0.0,
                fps=30.0
            )

            cv2.imshow("Jumping Jack Counter Demo", vis_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_idx += 1

        cap.release()

    cv2.destroyAllWindows()
    estimator.close()

    print(f"Final Jumping Jack Count: {counter.count}")


def main():
    """Main entry point for demo."""
    parser = argparse.ArgumentParser(description="Demo script for Tensor Push Up")
    parser.add_argument("--mode", type=str, choices=["pose", "pushup", "jumping_jack"],
                       default="pose", help="Demo mode to run")
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
