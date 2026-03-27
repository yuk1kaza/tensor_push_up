"""
Training Script Module

This module provides training functionality for the action classification model,
including data loading, training loop, callbacks, and model checkpointing.
"""

import os
import sys
import argparse
import json
import logging
import platform
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    create_model, compile_model, create_callbacks,
    load_model_from_checkpoint, export_model,
    ACTION_CLASSES, ACTION_LABELS
)
from src.utils import (
    setup_logging, ensure_dir, Timer, ProgressTracker
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for training action classification models.

    Handles:
    1. Loading configuration from YAML
    2. Loading training/validation data
    3. Building and compiling the model
    4. Training with callbacks
    5. Saving checkpoints and exporting models
    6. Training history tracking
    """

    def __init__(self, config_path: str = "configs/train.yaml"):
        """
        Initialize the trainer.

        Args:
            config_path: Path to training configuration file
        """
        # Load configuration
        from src.utils import load_config
        self.config = load_config(config_path)

        # Setup logging
        log_dir = self.config.get('paths', {}).get('log_dir', 'logs')
        setup_logging(log_dir=log_dir)

        # Extract configuration
        self.model_config = self.config.get('model', {})
        self.data_config = self.config.get('data', {})
        self.training_config = self.config.get('training', {})
        self.paths_config = self.config.get('paths', {})

        # Set paths
        self.data_dir = self.paths_config.get('data_dir', 'data/processed')
        self.checkpoint_dir = self.paths_config.get('checkpoint_dir', 'models/checkpoints')
        self.export_dir = self.paths_config.get('export_dir', 'models/exported')
        self.log_dir = log_dir

        # Ensure directories exist
        ensure_dir(self.checkpoint_dir)
        ensure_dir(self.export_dir)
        ensure_dir(self.log_dir)

        # Initialize model and datasets
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.history = None
        self.input_shape = None
        self.available_label_ids = []

        self._log_runtime_environment()

        logger.info("Trainer initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
        logger.info(f"Export directory: {self.export_dir}")

    def _log_runtime_environment(self):
        """Log TensorFlow runtime details, especially Windows/WSL GPU constraints."""
        gpus = tf.config.list_physical_devices('GPU')
        gpu_names = [gpu.name for gpu in gpus]

        logger.info(
            "Runtime environment: platform=%s, release=%s, tensorflow=%s, gpus=%s",
            sys.platform,
            platform.release(),
            tf.__version__,
            gpu_names or "[]"
        )

        release = platform.release().lower()
        if 'microsoft' in release or 'wsl' in release:
            logger.info("WSL environment detected. GPU training depends on WSL CUDA passthrough and Linux CUDA setup.")
        elif sys.platform.startswith("win") and not gpus:
            logger.warning(
                "Native Windows TensorFlow is running without GPU devices. "
                "For NVIDIA GPU training, use WSL2 and run this same command from the Linux environment."
            )

    def _validate_feature_shapes(
        self,
        train_features: np.ndarray,
        val_features: np.ndarray,
        test_features: np.ndarray
    ):
        """Validate feature tensor ranks and keep the actual dataset shape."""
        datasets = {
            'train': train_features,
            'val': val_features,
            'test': test_features
        }

        for dataset_name, features in datasets.items():
            if features.ndim != 3:
                raise ValueError(
                    f"{dataset_name}_features.npy must be a 3D tensor of shape "
                    f"(samples, window_size, feature_dim), got {features.shape}."
                )

        actual_input_shape = tuple(int(dim) for dim in train_features.shape[1:])
        for dataset_name, features in datasets.items():
            if tuple(int(dim) for dim in features.shape[1:]) != actual_input_shape:
                raise ValueError(
                    f"Feature shape mismatch: train uses {actual_input_shape} but "
                    f"{dataset_name} uses {features.shape[1:]}. Re-run preprocessing."
                )

        config_input_shape = tuple(self.model_config.get('input_shape', []))
        config_window_size = self.data_config.get('window_size')

        if config_input_shape and tuple(int(dim) for dim in config_input_shape) != actual_input_shape:
            logger.warning(
                "Config input_shape %s does not match dataset shape %s. "
                "Training will use the dataset shape.",
                config_input_shape,
                actual_input_shape
            )

        if config_window_size and int(config_window_size) != actual_input_shape[0]:
            logger.warning(
                "Config window_size=%s does not match dataset window size=%s. "
                "Training will use the dataset window size.",
                config_window_size,
                actual_input_shape[0]
            )

        self.input_shape = actual_input_shape
        self.model_config['input_shape'] = list(actual_input_shape)
        self.data_config['window_size'] = actual_input_shape[0]

    def _validate_label_distribution(
        self,
        train_labels: np.ndarray,
        val_labels: np.ndarray,
        test_labels: np.ndarray
    ):
        """Validate label coverage and fail fast on unusable training sets."""
        combined_labels = np.concatenate([train_labels, val_labels, test_labels], axis=0)
        unique_labels = sorted(int(label) for label in np.unique(combined_labels))
        self.available_label_ids = unique_labels

        configured_num_classes = int(self.model_config.get('num_classes', 3))
        max_label_id = max(unique_labels) if unique_labels else -1
        if max_label_id >= configured_num_classes:
            raise ValueError(
                f"Found label id {max_label_id}, but num_classes={configured_num_classes}. "
                f"Update configs/train.yaml or regenerate labels."
            )

        if len(unique_labels) < 2:
            action_name = ACTION_CLASSES.get(unique_labels[0], 'unknown') if unique_labels else 'none'
            message = (
                "Only one class is present in the processed dataset "
                f"({action_name}). This training run would be misleading and usually "
                "means data/labels is empty or preprocessing assigned every sample to 'other'. "
                "Create label JSON files under data/labels, re-run preprocess.py, then train again."
            )
            if self.training_config.get('allow_single_class', False):
                logger.warning("%s Continuing only because allow_single_class=true.", message)
            else:
                raise ValueError(message)

        if len(unique_labels) < configured_num_classes:
            missing_labels = [
                ACTION_CLASSES.get(label_id, f"class_{label_id}")
                for label_id in range(configured_num_classes)
                if label_id not in unique_labels
            ]
            logger.warning(
                "Dataset currently contains %s/%s classes. Missing classes: %s",
                len(unique_labels),
                configured_num_classes,
                missing_labels
            )

    def load_data(self):
        """Load training, validation, and test data."""
        data_dir = self.data_dir
        batch_size = self.data_config.get('batch_size', 32)
        shuffle = self.data_config.get('shuffle', True)

        # Check if split files exist
        required_files = [
            os.path.join(data_dir, 'train_features.npy'),
            os.path.join(data_dir, 'train_labels.npy'),
            os.path.join(data_dir, 'val_features.npy'),
            os.path.join(data_dir, 'val_labels.npy')
        ]

        for file_path in required_files:
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Data file not found: {file_path}. "
                    f"Please run preprocess.py first to generate training data."
                )

        # Load data
        logger.info("Loading training data...")
        train_features = np.load(os.path.join(data_dir, 'train_features.npy')).astype(np.float32)
        train_labels = np.load(os.path.join(data_dir, 'train_labels.npy')).astype(np.int32)

        logger.info("Loading validation data...")
        val_features = np.load(os.path.join(data_dir, 'val_features.npy')).astype(np.float32)
        val_labels = np.load(os.path.join(data_dir, 'val_labels.npy')).astype(np.int32)

        # Load test data if available
        test_file = os.path.join(data_dir, 'test_features.npy')
        if os.path.exists(test_file):
            logger.info("Loading test data...")
            test_features = np.load(test_file).astype(np.float32)
            test_labels = np.load(os.path.join(data_dir, 'test_labels.npy')).astype(np.int32)
        else:
            logger.warning("Test data not found, using validation data")
            test_features, test_labels = val_features, val_labels

        self._validate_feature_shapes(train_features, val_features, test_features)
        self._validate_label_distribution(train_labels, val_labels, test_labels)

        # Log dataset statistics
        logger.info(f"Train samples: {len(train_features)}")
        logger.info(f"Val samples: {len(val_features)}")
        logger.info(f"Test samples: {len(test_features)}")
        logger.info(f"Actual dataset input shape: {self.input_shape}")

        # Print label distribution
        for dataset_name, labels in [
            ('train', train_labels),
            ('val', val_labels),
            ('test', test_labels)
        ]:
            unique, counts = np.unique(labels, return_counts=True)
            dist = {ACTION_CLASSES.get(int(i), 'unknown'): int(c) for i, c in zip(unique, counts)}
            logger.info(f"{dataset_name.capitalize()} distribution: {dist}")

        # Create TensorFlow datasets
        def create_dataset(features, labels, is_train=True):
            dataset = tf.data.Dataset.from_tensor_slices((features, labels))

            if is_train and shuffle:
                dataset = dataset.shuffle(buffer_size=10000, seed=42)

            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset

        self.train_dataset = create_dataset(train_features, train_labels, is_train=True)
        self.val_dataset = create_dataset(val_features, val_labels, is_train=False)
        self.test_dataset = create_dataset(test_features, test_labels, is_train=False)

        # Save dataset info
        dataset_info = {
            'train_samples': len(train_features),
            'val_samples': len(val_features),
            'test_samples': len(test_features),
            'batch_size': batch_size,
            'input_shape': list(self.input_shape) if self.input_shape is not None else None,
            'available_label_ids': self.available_label_ids,
            'available_labels': [
                ACTION_CLASSES.get(label_id, f"class_{label_id}")
                for label_id in self.available_label_ids
            ],
            'train_distribution': {
                ACTION_CLASSES.get(int(i), 'unknown'): int(c)
                for i, c in zip(*np.unique(train_labels, return_counts=True))
            },
            'val_distribution': {
                ACTION_CLASSES.get(int(i), 'unknown'): int(c)
                for i, c in zip(*np.unique(val_labels, return_counts=True))
            },
            'test_distribution': {
                ACTION_CLASSES.get(int(i), 'unknown'): int(c)
                for i, c in zip(*np.unique(test_labels, return_counts=True))
            }
        }

        with open(os.path.join(self.log_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)

        logger.info("Data loaded successfully")

    def build_model(self, resume_from: str = None):
        """
        Build the model.

        Args:
            resume_from: Path to checkpoint to resume from
        """
        if self.input_shape is not None:
            input_shape = self.input_shape
        else:
            input_shape = tuple(self.model_config.get('input_shape', [30, 32]))

        num_classes = self.model_config.get('num_classes', 3)

        logger.info(f"Building model with input shape: {input_shape}")
        logger.info(f"Number of classes: {num_classes}")

        # Load from checkpoint if resuming
        if resume_from and os.path.exists(resume_from):
            logger.info(f"Loading model from checkpoint: {resume_from}")
            self.model = load_model_from_checkpoint(resume_from)
        else:
            # Create new model
            self.model = create_model(
                model_type=self.model_config.get('name', 'lstm'),
                input_shape=input_shape,
                num_classes=num_classes,
                lstm_units=self.model_config.get('lstm_units', [128, 64]),
                dropout_rate=self.model_config.get('dropout_rate', 0.5),
                dense_units=self.model_config.get('dense_units', [32])
            )

        # Compile model
        learning_rate = self.model_config.get('learning_rate', 0.001)
        self.model = compile_model(
            self.model,
            learning_rate=learning_rate,
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Print model summary
        self.model.summary()

    def train(self, epochs: int = None):
        """
        Train the model.

        Args:
            epochs: Number of epochs (overrides config)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        if self.train_dataset is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Get training parameters
        epochs = epochs or self.training_config.get('epochs', 100)

        # Create callbacks
        tensorboard_log_dir = os.path.join(
            self.log_dir,
            self.training_config.get('tensorboard_log_dir', 'tensorboard')
        )

        callbacks_list = create_callbacks(
            checkpoint_dir=self.checkpoint_dir,
            tensorboard_log_dir=tensorboard_log_dir,
            early_stopping_patience=self.training_config.get('early_stopping_patience', 15),
            reduce_lr_patience=self.training_config.get('reduce_lr_patience', 5),
            checkpoint_monitor=self.training_config.get('checkpoint_monitor', 'val_loss'),
            use_tensorboard=self.training_config.get('use_tensorboard', True)
        )

        # Train model
        logger.info(f"Starting training for {epochs} epochs...")

        with Timer() as timer:
            self.history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=epochs,
                callbacks=callbacks_list,
                verbose=1
            )

        logger.info(f"Training completed in {timer.elapsed():.2f} seconds")

        # Save training history
        self.save_history()

        # Evaluate on test set
        self.evaluate()

    def evaluate(self):
        """Evaluate the model on the test dataset."""
        if self.model is None or self.test_dataset is None:
            logger.warning("Cannot evaluate: model or test data not available")
            return

        logger.info("Evaluating model on test set...")

        # Get predictions
        predictions = self.model.predict(self.test_dataset, verbose=0)
        predicted_labels = np.argmax(predictions, axis=1)

        # Get true labels
        true_labels = np.concatenate([y for _, y in self.test_dataset], axis=0)

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, confusion_matrix, classification_report
        )

        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

        evaluated_label_ids = sorted(
            set(int(label) for label in np.unique(true_labels))
            | set(int(label) for label in np.unique(predicted_labels))
        )
        target_names = [ACTION_CLASSES.get(label_id, f"class_{label_id}") for label_id in evaluated_label_ids]

        if len(evaluated_label_ids) < int(self.model_config.get('num_classes', 3)):
            logger.warning(
                "Evaluation is running on a subset of classes only: %s",
                target_names
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="A single label was found in 'y_true' and 'y_pred'.*"
            )

            # Per-class metrics
            per_class_report = classification_report(
                true_labels, predicted_labels,
                labels=evaluated_label_ids,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )

            # Confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=evaluated_label_ids)

        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'evaluated_label_ids': evaluated_label_ids,
            'evaluated_labels': target_names,
            'per_class_metrics': per_class_report,
            'confusion_matrix': cm.tolist()
        }

        # Save metrics
        with open(os.path.join(self.log_dir, 'test_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")

        # Print confusion matrix
        logger.info("Confusion Matrix:")
        for i, row in enumerate(cm):
            logger.info(f"  {target_names[i]}: {row}")

        return metrics

    def save_history(self):
        """Save training history to file."""
        if self.history is None:
            return

        history_data = {
            'loss': [float(x) for x in self.history.history.get('loss', [])],
            'accuracy': [float(x) for x in self.history.history.get('accuracy', [])],
            'val_loss': [float(x) for x in self.history.history.get('val_loss', [])],
            'val_accuracy': [float(x) for x in self.history.history.get('val_accuracy', [])]
        }

        with open(os.path.join(self.log_dir, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info("Training history saved")

    def export(self, formats: list = None):
        """
        Export the trained model.

        Args:
            formats: List of export formats ('saved_model', 'h5', 'tflite', 'onnx')
        """
        if self.model is None:
            raise ValueError("No model to export. Train the model first.")

        if formats is None:
            formats = ['saved_model', 'h5']

        # Load best checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.keras')

        if os.path.exists(checkpoint_path):
            logger.info(f"Loading best model from {checkpoint_path}")
            self.model = load_model_from_checkpoint(checkpoint_path)

        export_model(self.model, self.export_dir, 'action_classifier', formats)

        # Save model summary
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))

        with open(os.path.join(self.export_dir, 'model_summary.txt'), 'w') as f:
            f.write('\n'.join(summary))

        logger.info(f"Model exported to {self.export_dir}")

    def run(self, resume_from: str = None, export: bool = True, epochs: int = None):
        """
        Run the complete training pipeline.

        Args:
            resume_from: Path to checkpoint to resume from
            export: Whether to export the model after training
            epochs: Number of epochs to train for (overrides config)
        """
        logger.info("Starting training pipeline...")

        # Load data
        self.load_data()

        # Build model
        self.build_model(resume_from=resume_from)

        # Train
        self.train(epochs=epochs)

        # Export
        if export:
            self.export()

        logger.info("Training pipeline complete!")


def main():
    """Main entry point for training."""
    parser = argparse.ArgumentParser(description="Train action classification model")
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to training configuration file")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of training epochs (overrides config)")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip model export after training")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Learning rate (overrides config)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (overrides config)")
    parser.add_argument("--allow-single-class", action="store_true",
                        help="Allow training/evaluation to continue even if the dataset only contains one class")

    args = parser.parse_args()

    # Initialize trainer
    trainer = Trainer(config_path=args.config)

    # Override config with command-line arguments
    if args.batch_size:
        trainer.data_config['batch_size'] = args.batch_size
        logger.info(f"Overriding batch size to {args.batch_size}")

    if args.learning_rate:
        trainer.model_config['learning_rate'] = args.learning_rate
        logger.info(f"Overriding learning rate to {args.learning_rate}")

    if args.data_dir:
        trainer.data_dir = args.data_dir
        logger.info(f"Overriding data directory to {args.data_dir}")

    if args.allow_single_class:
        trainer.training_config['allow_single_class'] = True
        logger.warning("allow_single_class enabled. This should only be used for smoke tests.")

    # Run training
    trainer.run(
        resume_from=args.resume,
        export=not args.no_export,
        epochs=args.epochs
    )


if __name__ == "__main__":
    main()
