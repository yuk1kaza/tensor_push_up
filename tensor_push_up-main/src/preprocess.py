"""
Data Preprocessing Module

This module handles data preprocessing for the Tensor Push Up project,
including video processing, feature extraction, data augmentation, and
sliding window construction for training data preparation.
"""

import os
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from .pose_estimator import PoseEstimator
from .utils import (
    ensure_dir, get_files_by_extension, split_train_val_test,
    Timer, ProgressTracker, setup_logging
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocessor for extracting pose features from raw video data.

    This class handles:
    1. Loading videos from raw directory
    2. Extracting pose keypoints using MediaPipe
    3. Calculating joint angles
    4. Applying data augmentation
    5. Creating sliding window samples
    6. Saving processed data in structured format
    """

    def __init__(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
        labels_dir: str = "data/labels",
        window_size: int = 30,
        stride: int = 5,
        video_fps: int = 30,
        target_resolution: Tuple[int, int] = (1280, 720),
        min_pose_confidence: float = 0.5
    ):
        """
        Initialize the data preprocessor.

        Args:
            input_dir: Directory containing raw videos
            output_dir: Directory to save processed data
            labels_dir: Directory containing label files
            window_size: Sliding window size for sequences
            stride: Stride for sliding windows
            video_fps: Target FPS for video processing
            target_resolution: Target resolution for frame resizing
            min_pose_confidence: Minimum pose confidence threshold
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.labels_dir = labels_dir
        self.window_size = window_size
        self.stride = stride
        self.video_fps = video_fps
        self.target_resolution = target_resolution
        self.min_pose_confidence = min_pose_confidence

        # Create output directories
        ensure_dir(output_dir)
        ensure_dir(os.path.join(output_dir, "features"))
        ensure_dir(os.path.join(output_dir, "samples"))
        ensure_dir(os.path.join(output_dir, "metadata"))

        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            min_detection_confidence=min_pose_confidence,
            min_tracking_confidence=min_pose_confidence
        )

        # Video extensions to process
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

        logger.info(f"DataPreprocessor initialized:")
        logger.info(f"  Input dir: {input_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Window size: {window_size}, Stride: {stride}")

    def load_labels(self) -> Dict[str, Dict]:
        """
        Load label files from labels directory.

        Expected label format (JSON):
        {
            "video_name.mp4": {
                "action_type": "pushup" | "jumping_jack" | "other",
                "count": 10,
                "start_frame": 0,
                "end_frame": 300,
                "phases": [
                    {"frame": 0, "phase": "start"},
                    {"frame": 150, "phase": "low"},
                    {"frame": 300, "phase": "end"}
                ]
            }
        }

        Returns:
            Dictionary mapping video names to labels
        """
        labels = {}

        if not os.path.exists(self.labels_dir):
            logger.warning(f"Labels directory not found: {self.labels_dir}")
            return labels

        # Find all JSON files in labels directory
        label_files = get_files_by_extension(self.labels_dir, ['.json'], recursive=True)

        for label_file in label_files:
            try:
                with open(label_file, 'r', encoding='utf-8') as f:
                    file_labels = json.load(f)

                    # Merge labels
                    labels.update(file_labels)

            except Exception as e:
                logger.error(f"Error loading labels from {label_file}: {e}")

        logger.info(f"Loaded labels for {len(labels)} videos")
        return labels

    def extract_features_from_video(
        self,
        video_path: str,
        label: Optional[Dict] = None,
        augment: bool = False
    ) -> Dict[str, any]:
        """
        Extract pose features from a single video.

        Args:
            video_path: Path to video file
            label: Optional label dictionary for this video
            augment: Whether to apply data augmentation

        Returns:
            Dictionary containing extracted features and metadata
        """
        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]

        logger.debug(f"Processing video: {video_path}")

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {'video_id': video_id, 'error': 'Failed to open video'}

        # Get video properties
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine frame sampling
        if original_fps != self.video_fps:
            frame_skip = max(1, original_fps // self.video_fps)
        else:
            frame_skip = 1

        # Extract frames
        frames = []
        keypoints_list = []
        angles_list = []
        features_list = []
        valid_frames = []

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample frames at target FPS
            if frame_idx % frame_skip == 0:
                # Resize if needed
                if (width, height) != self.target_resolution:
                    frame = cv2.resize(frame, self.target_resolution)
                    width, height = self.target_resolution

                # Extract pose
                keypoints, angles = self.pose_estimator.process_frame(frame)

                if keypoints is not None and angles is not None:
                    # Extract features
                    features = self.pose_estimator.extract_features(keypoints, angles)

                    frames.append(frame)
                    keypoints_list.append(keypoints)
                    angles_list.append(angles)
                    features_list.append(features)
                    valid_frames.append(frame_idx)

            frame_idx += 1

        cap.release()

        if len(features_list) == 0:
            logger.warning(f"No valid poses detected in {video_name}")
            return {
                'video_id': video_id,
                'error': 'No valid poses detected',
                'total_frames': frame_idx
            }

        # Convert to numpy arrays
        features_array = np.array(features_list)
        keypoints_array = np.array(keypoints_list)

        # Apply augmentation if requested
        if augment:
            features_array = self._augment_features(features_array)

        # Create sliding window samples
        samples = self._create_sliding_windows(
            features_array,
            label=label if label else None
        )

        # Apply temporal augmentation
        if augment:
            samples = self._temporal_augmentation(samples)

        result = {
            'video_id': video_id,
            'video_name': video_name,
            'features': features_array,
            'keypoints': keypoints_array,
            'samples': samples,
            'original_fps': original_fps,
            'sampled_fps': len(features_list) / (frame_idx / original_fps) if frame_idx > 0 else 0,
            'total_frames': frame_idx,
            'valid_frames': len(valid_frames),
            'label': label,
            'resolution': (width, height)
        }

        logger.info(f"Extracted {len(features_list)} frames from {video_name}, "
                   f"created {len(samples['features'])} samples")

        return result

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply spatial augmentation to features.

        Args:
            features: Feature array of shape (T, F)

        Returns:
            Augmented features
        """
        augmented = []

        # Horizontal flip (simulate mirror view)
        # For keypoints, this means swapping left/right landmarks
        flipped = features.copy()

        # Swap left/right keypoints (pairs of coordinates)
        # MediaPipe left landmarks: 11-22, 23-24, 25-26, 27-28
        # Right landmarks are the corresponding pairs
        # Since our features are already flattened and selected,
        # we need to swap the appropriate pairs
        swap_pairs = [
            (0, 1),  # left/right shoulders
            (2, 3),  # left/right hips
            (4, 5),  # left/right elbows
            (6, 7),  # left/right wrists
            (8, 9),  # left/right knees
            (10, 11) # left/right ankles
        ]

        for i, j in swap_pairs:
            flipped[:, [i, j]] = flipped[:, [j, i]]

        augmented.append(flipped)

        # Add small random noise
        noise = np.random.normal(0, 0.01, features.shape)
        noisy = features + noise
        augmented.append(noisy)

        return np.concatenate([features] + augmented, axis=0)

    def _temporal_augmentation(self, samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply temporal augmentation to samples.

        Args:
            samples: Dictionary containing sample arrays

        Returns:
            Augmented samples
        """
        augmented_features = []
        augmented_labels = []

        for features, label in zip(samples['features'], samples['labels']):
            augmented_features.append(features)
            augmented_labels.append(label)

            # Time stretching (speed variation)
            if len(features) == self.window_size:
                # Subsample to create "faster" version
                fast_indices = np.linspace(0, len(features) - 1, len(features), dtype=int)
                fast_features = features[fast_indices]
                augmented_features.append(fast_features)
                augmented_labels.append(label)

        return {
            'features': np.array(augmented_features),
            'labels': np.array(augmented_labels)
        }

    def _create_sliding_windows(
        self,
        features: np.ndarray,
        label: Optional[Dict] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create sliding window samples from features.

        Args:
            features: Feature array of shape (T, F)
            label: Optional label dictionary

        Returns:
            Dictionary with 'features' and 'labels' arrays
        """
        windows = []
        labels = []

        for i in range(0, len(features) - self.window_size + 1, self.stride):
            window = features[i:i + self.window_size]

            # Pad if necessary
            if len(window) < self.window_size:
                padding = np.zeros((self.window_size - len(window), features.shape[1]))
                window = np.vstack([window, padding])

            windows.append(window)

            # Determine label
            if label is not None:
                action_type = label.get('action_type', 'other')
                if action_type == 'pushup':
                    labels.append(0)
                elif action_type == 'jumping_jack':
                    labels.append(1)
                else:
                    labels.append(2)
            else:
                # Default to 'other' if no label
                labels.append(2)

        return {
            'features': np.array(windows),
            'labels': np.array(labels)
        }

    def process_videos(
        self,
        action_type: Optional[str] = None,
        augment: bool = False,
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, any]:
        """
        Process all videos in the input directory.

        Args:
            action_type: Optional action type filter
            augment: Whether to apply data augmentation
            parallel: Whether to use parallel processing
            max_workers: Number of parallel workers

        Returns:
            Dictionary containing all processed data
        """
        # Get all video files
        video_files = get_files_by_extension(
            self.input_dir,
            self.video_extensions,
            recursive=True
        )

        logger.info(f"Found {len(video_files)} video files to process")

        # Load labels
        labels = self.load_labels()

        # Filter by action type if specified
        if action_type:
            video_files = [
                f for f in video_files
                if labels.get(os.path.basename(f), {}).get('action_type') == action_type
            ]
            logger.info(f"Filtered to {len(video_files)} {action_type} videos")

        # Process videos
        all_results = []
        all_features = []
        all_labels = []

        if parallel and len(video_files) > 1:
            # Parallel processing
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}

                for video_file in video_files:
                    video_name = os.path.basename(video_file)
                    video_label = labels.get(video_name)

                    future = executor.submit(
                        self._process_single_video,
                        video_file,
                        video_label,
                        augment,
                        self.window_size,
                        self.stride,
                        self.video_fps,
                        self.target_resolution,
                        self.min_pose_confidence
                    )
                    futures[future] = video_file

                # Collect results
                with ProgressTracker(len(futures), "Processing videos") as pbar:
                    for future in as_completed(futures):
                        result = future.result()
                        all_results.append(result)

                        if 'samples' in result:
                            all_features.extend(result['samples']['features'])
                            all_labels.extend(result['samples']['labels'])

                        pbar.update(1)
        else:
            # Sequential processing
            with ProgressTracker(len(video_files), "Processing videos") as pbar:
                for video_file in video_files:
                    video_name = os.path.basename(video_file)
                    video_label = labels.get(video_name)

                    result = self.extract_features_from_video(
                        video_file,
                        label=video_label,
                        augment=augment
                    )
                    all_results.append(result)

                    if 'samples' in result:
                        all_features.extend(result['samples']['features'])
                        all_labels.extend(result['samples']['labels'])

                    pbar.update(1)

        # Combine all features
        if all_features:
            combined_data = {
                'features': np.array(all_features),
                'labels': np.array(all_labels),
                'metadata': all_results
            }

            # Save processed data
            self._save_processed_data(combined_data)

            return combined_data
        else:
            logger.warning("No features extracted from any videos")
            return {'features': np.array([]), 'labels': np.array([]), 'metadata': []}

    @staticmethod
    def _process_single_video(
        video_path: str,
        label: Optional[Dict],
        augment: bool,
        window_size: int,
        stride: int,
        video_fps: int,
        target_resolution: Tuple[int, int],
        min_pose_confidence: float
    ) -> Dict[str, any]:
        """
        Static method for processing a single video (for parallel processing).

        This is a workaround since PoseEstimator can't be pickled.
        """
        # Initialize pose estimator in subprocess
        pose_estimator = PoseEstimator(
            min_detection_confidence=min_pose_confidence,
            min_tracking_confidence=min_pose_confidence
        )

        video_name = os.path.basename(video_path)
        video_id = os.path.splitext(video_name)[0]

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'video_id': video_id, 'error': 'Failed to open video'}

        original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Determine frame sampling
        if original_fps != video_fps:
            frame_skip = max(1, original_fps // video_fps)
        else:
            frame_skip = 1

        features_list = []
        valid_frames = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                if (width, height) != target_resolution:
                    frame = cv2.resize(frame, target_resolution)

                keypoints, angles = pose_estimator.process_frame(frame)

                if keypoints is not None and angles is not None:
                    features = pose_estimator.extract_features(keypoints, angles)
                    features_list.append(features)
                    valid_frames.append(frame_idx)

            frame_idx += 1

        cap.release()
        pose_estimator.close()

        if len(features_list) == 0:
            return {'video_id': video_id, 'error': 'No valid poses detected'}

        features_array = np.array(features_list)

        # Create sliding windows
        windows = []
        labels = []

        for i in range(0, len(features_array) - window_size + 1, stride):
            window = features_array[i:i + window_size]

            if len(window) < window_size:
                padding = np.zeros((window_size - len(window), features_array.shape[1]))
                window = np.vstack([window, padding])

            windows.append(window)

            if label is not None:
                action_type = label.get('action_type', 'other')
                if action_type == 'pushup':
                    labels.append(0)
                elif action_type == 'jumping_jack':
                    labels.append(1)
                else:
                    labels.append(2)
            else:
                labels.append(2)

        return {
            'video_id': video_id,
            'video_name': video_name,
            'samples': {'features': np.array(windows), 'labels': np.array(labels)},
            'total_frames': frame_idx,
            'valid_frames': len(valid_frames),
            'label': label
        }

    def _save_processed_data(self, data: Dict[str, any]):
        """
        Save processed data to disk.

        Args:
            data: Dictionary containing features, labels, and metadata
        """
        # Save features and labels
        np.save(
            os.path.join(self.output_dir, "features.npy"),
            data['features']
        )
        np.save(
            os.path.join(self.output_dir, "labels.npy"),
            data['labels']
        )

        # Save samples
        np.save(
            os.path.join(self.output_dir, "samples", "features.npy"),
            data['features']
        )
        np.save(
            os.path.join(self.output_dir, "samples", "labels.npy"),
            data['labels']
        )

        # Save metadata as JSON
        metadata = []
        for item in data['metadata']:
            metadata_item = {
                'video_id': item.get('video_id'),
                'video_name': item.get('video_name'),
                'total_frames': item.get('total_frames'),
                'valid_frames': item.get('valid_frames'),
                'label': item.get('label'),
                'resolution': item.get('resolution')
            }
            if 'error' in item:
                metadata_item['error'] = item['error']

            metadata.append(metadata_item)

        with open(
            os.path.join(self.output_dir, "metadata", "metadata.json"),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Save dataset statistics
        stats = {
            'num_videos': len(data['metadata']),
            'num_samples': len(data['features']),
            'feature_shape': data['features'].shape,
            'label_distribution': {
                'pushup': int(np.sum(data['labels'] == 0)),
                'jumping_jack': int(np.sum(data['labels'] == 1)),
                'other': int(np.sum(data['labels'] == 2))
            }
        }

        with open(
            os.path.join(self.output_dir, "dataset_stats.json"),
            'w',
            encoding='utf-8'
        ) as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved processed data to {self.output_dir}")
        logger.info(f"  Samples: {stats['num_samples']}")
        logger.info(f"  Label distribution: {stats['label_distribution']}")

    def split_dataset(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        random_seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Split the dataset into train, validation, and test sets.

        Args:
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary with train/val/test features and labels
        """
        # Load processed data
        features_path = os.path.join(self.output_dir, "samples", "features.npy")
        labels_path = os.path.join(self.output_dir, "samples", "labels.npy")

        if not os.path.exists(features_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                "Processed data not found. Run process_videos() first."
            )

        features = np.load(features_path)
        labels = np.load(labels_path)

        # Get sample indices
        num_samples = len(features)
        indices = np.arange(num_samples)
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        # Calculate split points
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)

        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]

        # Create splits
        splits = {
            'train_features': features[train_indices],
            'train_labels': labels[train_indices],
            'val_features': features[val_indices],
            'val_labels': labels[val_indices],
            'test_features': features[test_indices],
            'test_labels': labels[test_indices]
        }

        # Save splits
        for name, data in splits.items():
            np.save(os.path.join(self.output_dir, f"{name}.npy"), data)

        # Save split info
        split_info = {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'train_samples': len(train_indices),
            'val_samples': len(val_indices),
            'test_samples': len(test_indices),
            'random_seed': random_seed
        }

        with open(
            os.path.join(self.output_dir, "split_info.json"),
            'w'
        ) as f:
            json.dump(split_info, f, indent=2)

        logger.info(f"Dataset split: Train={len(train_indices)}, "
                   f"Val={len(val_indices)}, Test={len(test_indices)}")

        return splits

    def create_tf_dataset(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        cache: bool = True
    ):
        """
        Create TensorFlow datasets from processed data.

        Args:
            batch_size: Batch size for datasets
            shuffle: Whether to shuffle training data
            cache: Whether to cache datasets in memory

        Returns:
            Dictionary of tf.data.Dataset objects
        """
        import tensorflow as tf

        # Load splits
        train_features = np.load(os.path.join(self.output_dir, "train_features.npy"))
        train_labels = np.load(os.path.join(self.output_dir, "train_labels.npy"))
        val_features = np.load(os.path.join(self.output_dir, "val_features.npy"))
        val_labels = np.load(os.path.join(self.output_dir, "val_labels.npy"))
        test_features = np.load(os.path.join(self.output_dir, "test_features.npy"))
        test_labels = np.load(os.path.join(self.output_dir, "test_labels.npy"))

        # Create datasets
        def create_dataset(features, labels, name):
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))

            if shuffle and name == 'train':
                dataset = dataset.shuffle(buffer_size=10000)

            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            if cache:
                dataset = dataset.cache()

            return dataset

        datasets = {
            'train': create_dataset(train_features, train_labels, 'train'),
            'val': create_dataset(val_features, val_labels, 'val'),
            'test': create_dataset(test_features, test_labels, 'test')
        }

        logger.info(f"Created TensorFlow datasets with batch_size={batch_size}")

        return datasets

    def close(self):
        """Release resources."""
        if hasattr(self, 'pose_estimator'):
            self.pose_estimator.close()


def create_sample_labels(output_dir: str = "data/labels"):
    """
    Create sample label files for demonstration.

    Args:
        output_dir: Directory to save sample labels
    """
    ensure_dir(output_dir)

    sample_labels = {
        "sample_pushup_01.mp4": {
            "action_type": "pushup",
            "count": 10,
            "start_frame": 0,
            "end_frame": 300,
            "notes": "Standard push-ups at moderate pace"
        },
        "sample_pushup_02.mp4": {
            "action_type": "pushup",
            "count": 5,
            "start_frame": 0,
            "end_frame": 150,
            "notes": "Fast push-ups"
        },
        "sample_jumping_jack_01.mp4": {
            "action_type": "jumping_jack",
            "count": 15,
            "start_frame": 0,
            "end_frame": 300,
            "notes": "Standard jumping jacks"
        }
    }

    with open(os.path.join(output_dir, "sample_labels.json"), 'w') as f:
        json.dump(sample_labels, f, indent=2)

    logger.info(f"Created sample labels in {output_dir}")


def main():
    """Main entry point for data preprocessing."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess video data for action recognition")
    parser.add_argument("--input", type=str, default="data/raw", help="Input directory with raw videos")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory for processed data")
    parser.add_argument("--labels", type=str, default="data/labels", help="Directory with label files")
    parser.add_argument("--action", type=str, choices=["pushup", "jumping_jack", "all"], default="all",
                        help="Action type to process")
    parser.add_argument("--window-size", type=int, default=30, help="Sliding window size")
    parser.add_argument("--stride", type=int, default=5, help="Sliding window stride")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")
    parser.add_argument("--create-sample-labels", action="store_true", help="Create sample label files")

    args = parser.parse_args()

    # Create sample labels if requested
    if args.create_sample_labels:
        create_sample_labels(args.labels)

    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        input_dir=args.input,
        output_dir=args.output,
        labels_dir=args.labels,
        window_size=args.window_size,
        stride=args.stride
    )

    # Process videos
    action_type = None if args.action == "all" else args.action

    with Timer() as timer:
        data = preprocessor.process_videos(
            action_type=action_type,
            augment=args.augment,
            parallel=not args.no_parallel
        )

    logger.info(f"Processing completed in {timer.elapsed():.2f} seconds")

    # Split dataset
    splits = preprocessor.split_dataset()

    # Create TF datasets
    datasets = preprocessor.create_tf_dataset()

    # Cleanup
    preprocessor.close()

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
