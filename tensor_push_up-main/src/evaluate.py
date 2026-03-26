"""
Evaluation Script Module

This module provides comprehensive evaluation functionality for the action
classification model, including classification metrics, count accuracy,
confusion matrix, ROC curves, and evaluation report generation.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    roc_auc_score
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import (
    load_model_from_checkpoint, ACTION_CLASSES,
    ModelInference
)
from src.utils import (
    setup_logging, ensure_dir, calculate_count_metrics,
    Timer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for action classification models.

    Handles:
    1. Loading trained model and test data
    2. Computing classification metrics
    3. Generating confusion matrix and ROC curves
    4. Computing count-based metrics
    5. Generating comprehensive evaluation reports
    """

    def __init__(
        self,
        model_path: str,
        data_dir: str = "data/processed",
        output_dir: str = "results/evaluation"
    ):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to trained model checkpoint
            data_dir: Directory containing test data
            output_dir: Directory to save evaluation results
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Setup logging
        setup_logging()
        ensure_dir(output_dir)

        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = load_model_from_checkpoint(model_path)

        # Initialize inference wrapper
        self.inference = ModelInference(
            model_path=model_path,
            window_size=30,
            smoothing_window=5,
            confidence_threshold=0.6
        )

        # Load test data
        self._load_test_data()

        # Prediction results
        self.predictions = None
        self.predicted_labels = None
        self.predicted_probs = None

        logger.info("ModelEvaluator initialized")

    def _load_test_data(self):
        """Load test data."""
        test_features_path = os.path.join(self.data_dir, "test_features.npy")
        test_labels_path = os.path.join(self.data_dir, "test_labels.npy")

        if not os.path.exists(test_features_path):
            # Try validation data if test data doesn't exist
            test_features_path = os.path.join(self.data_dir, "val_features.npy")
            test_labels_path = os.path.join(self.data_dir, "val_labels.npy")
            logger.info("Using validation data as test data")

        self.test_features = np.load(test_features_path)
        self.test_labels = np.load(test_labels_path)

        logger.info(f"Loaded {len(self.test_features)} test samples")

    def predict(self):
        """Generate predictions on test data."""
        logger.info("Generating predictions...")

        with Timer() as timer:
            self.predictions = self.model.predict(self.test_features, verbose=0)
            self.predicted_labels = np.argmax(self.predictions, axis=1)
            self.predicted_probs = self.predictions

        logger.info(f"Predictions generated in {timer.elapsed():.2f} seconds")

    def compute_classification_metrics(self) -> Dict:
        """
        Compute classification metrics.

        Returns:
            Dictionary of classification metrics
        """
        if self.predicted_labels is None:
            self.predict()

        logger.info("Computing classification metrics...")

        # Overall metrics
        accuracy = accuracy_score(self.test_labels, self.predicted_labels)
        precision_macro = precision_score(
            self.test_labels, self.predicted_labels,
            average='macro', zero_division=0
        )
        recall_macro = recall_score(
            self.test_labels, self.predicted_labels,
            average='macro', zero_division=0
        )
        f1_macro = f1_score(
            self.test_labels, self.predicted_labels,
            average='macro', zero_division=0
        )

        # Weighted metrics
        precision_weighted = precision_score(
            self.test_labels, self.predicted_labels,
            average='weighted', zero_division=0
        )
        recall_weighted = recall_score(
            self.test_labels, self.predicted_labels,
            average='weighted', zero_division=0
        )
        f1_weighted = f1_score(
            self.test_labels, self.predicted_labels,
            average='weighted', zero_division=0
        )

        # Per-class metrics
        per_class_report = classification_report(
            self.test_labels,
            self.predicted_labels,
            target_names=[ACTION_CLASSES.get(i, 'unknown') for i in range(3)],
            output_dict=True,
            zero_division=0
        )

        metrics = {
            'overall': {
                'accuracy': float(accuracy),
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted)
            },
            'per_class': per_class_report
        }

        logger.info(f"Classification metrics:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  F1 (macro): {f1_macro:.4f}")
        logger.info(f"  F1 (weighted): {f1_weighted:.4f}")

        return metrics

    def compute_confusion_matrix(self) -> np.ndarray:
        """
        Compute confusion matrix.

        Returns:
            Confusion matrix as numpy array
        """
        if self.predicted_labels is None:
            self.predict()

        cm = confusion_matrix(self.test_labels, self.predicted_labels)
        logger.info(f"Confusion matrix:\n{cm}")

        return cm

    def plot_confusion_matrix(
        self,
        cm: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ):
        """
        Plot and save confusion matrix.

        Args:
            cm: Confusion matrix (computed if None)
            save_path: Path to save plot (default: output_dir/confusion_matrix.png)
            show_plot: Whether to display the plot
        """
        if cm is None:
            cm = self.compute_confusion_matrix()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "confusion_matrix.png")

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[ACTION_CLASSES.get(i, 'unknown') for i in range(3)],
            yticklabels=[ACTION_CLASSES.get(i, 'unknown') for i in range(3)]
        )

        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()

        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def compute_roc_curves(self) -> Dict:
        """
        Compute ROC curves for each class.

        Returns:
            Dictionary with ROC data for each class
        """
        if self.predicted_probs is None:
            self.predict()

        logger.info("Computing ROC curves...")

        # Convert labels to one-hot
        num_classes = len(ACTION_CLASSES)
        test_labels_onehot = np.zeros((len(self.test_labels), num_classes))
        for i, label in enumerate(self.test_labels):
            test_labels_onehot[i, label] = 1

        # Compute ROC for each class
        roc_data = {}

        for class_id in range(num_classes):
            class_name = ACTION_CLASSES.get(class_id, f'class_{class_id}')

            fpr, tpr, _ = roc_curve(
                test_labels_onehot[:, class_id],
                self.predicted_probs[:, class_id]
            )
            roc_auc = auc(fpr, tpr)

            roc_data[class_name] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'auc': float(roc_auc)
            }

            logger.info(f"{class_name} AUC: {roc_auc:.4f}")

        return roc_data

    def plot_roc_curves(
        self,
        roc_data: Optional[Dict] = None,
        save_path: Optional[str] = None,
        show_plot: bool = False
    ):
        """
        Plot and save ROC curves.

        Args:
            roc_data: ROC data (computed if None)
            save_path: Path to save plot (default: output_dir/roc_curves.png)
            show_plot: Whether to display the plot
        """
        if roc_data is None:
            roc_data = self.compute_roc_curves()

        if save_path is None:
            save_path = os.path.join(self.output_dir, "roc_curves.png")

        # Create figure
        plt.figure(figsize=(10, 8))

        # Plot ROC curve for each class
        colors = ['blue', 'red', 'green']
        for i, (class_name, data) in enumerate(roc_data.items()):
            plt.plot(
                data['fpr'],
                data['tpr'],
                color=colors[i % len(colors)],
                lw=2,
                label=f"{class_name} (AUC = {data['auc']:.3f})"
            )

        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def evaluate_counting_accuracy(
        self,
        ground_truth_counts: List[int],
        predicted_counts: List[int],
        tolerance: int = 1
    ) -> Dict:
        """
        Evaluate counting accuracy.

        Args:
            ground_truth_counts: List of ground truth counts
            predicted_counts: List of predicted counts
            tolerance: Allowed absolute error for "correct" counts

        Returns:
            Dictionary of count-based metrics
        """
        metrics = calculate_count_metrics(
            predicted_counts,
            ground_truth_counts,
            tolerance=tolerance
        )

        logger.info(f"Count metrics:")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        logger.info(f"  Count Accuracy (±{tolerance}): {metrics['count_accuracy']:.2%}")
        logger.info(f"  Exact Accuracy: {metrics['exact_accuracy']:.2%}")

        return metrics

    def plot_training_history(
        self,
        history_path: str = "logs/training_history.json",
        save_path: Optional[str] = None,
        show_plot: bool = False
    ):
        """
        Plot training history from saved JSON file.

        Args:
            history_path: Path to training history JSON file
            save_path: Path to save plot (default: output_dir/training_history.png)
            show_plot: Whether to display the plot
        """
        if not os.path.exists(history_path):
            logger.warning(f"Training history not found at {history_path}")
            return

        with open(history_path, 'r') as f:
            history = json.load(f)

        if save_path is None:
            save_path = os.path.join(self.output_dir, "training_history.png")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        ax1.plot(history.get('loss', []), label='Training Loss', linewidth=2)
        ax1.plot(history.get('val_loss', []), label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot accuracy
        ax2.plot(history.get('accuracy', []), label='Training Accuracy', linewidth=2)
        ax2.plot(history.get('val_accuracy', []), label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history plot saved to {save_path}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive evaluation report.

        Args:
            save_path: Path to save report (default: output_dir/evaluation_report.txt)

        Returns:
            Report as string
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "evaluation_report.txt")

        # Generate all metrics
        classification_metrics = self.compute_classification_metrics()
        cm = self.compute_confusion_matrix()
        roc_data = self.compute_roc_curves()

        # Build report
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("ACTION CLASSIFICATION MODEL EVALUATION REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        report_lines.append(f"Model Path: {self.model_path}")
        report_lines.append(f"Test Samples: {len(self.test_features)}")
        report_lines.append("")
        report_lines.append("-" * 70)
        report_lines.append("CLASSIFICATION METRICS")
        report_lines.append("-" * 70)
        report_lines.append("")

        # Overall metrics
        overall = classification_metrics['overall']
        report_lines.append("Overall Metrics:")
        report_lines.append(f"  Accuracy:          {overall['accuracy']:.4f}")
        report_lines.append(f"  Precision (macro): {overall['precision_macro']:.4f}")
        report_lines.append(f"  Recall (macro):    {overall['recall_macro']:.4f}")
        report_lines.append(f"  F1 Score (macro):  {overall['f1_macro']:.4f}")
        report_lines.append(f"  Precision (weighted): {overall['precision_weighted']:.4f}")
        report_lines.append(f"  Recall (weighted):    {overall['recall_weighted']:.4f}")
        report_lines.append(f"  F1 Score (weighted):  {overall['f1_weighted']:.4f}")
        report_lines.append("")

        # Per-class metrics
        report_lines.append("Per-Class Metrics:")
        for class_name in ['pushup', 'jumping_jack', 'other']:
            if class_name in classification_metrics['per_class']:
                class_metrics = classification_metrics['per_class'][class_name]
                report_lines.append(f"  {class_name.upper()}:")
                report_lines.append(f"    Precision: {class_metrics.get('precision', 0):.4f}")
                report_lines.append(f"    Recall:    {class_metrics.get('recall', 0):.4f}")
                report_lines.append(f"    F1 Score:  {class_metrics.get('f1-score', 0):.4f}")
                report_lines.append(f"    Support:   {int(class_metrics.get('support', 0))}")
        report_lines.append("")

        # Confusion matrix
        report_lines.append("-" * 70)
        report_lines.append("CONFUSION MATRIX")
        report_lines.append("-" * 70)
        report_lines.append("")
        report_lines.append("                Predicted")
        report_lines.append("            Pushup  J.Jack  Other")
        for i, row in enumerate(cm):
            true_label = ACTION_CLASSES.get(i, 'unknown').upper()[:8]
            report_lines.append(f"True {true_label:>2}  {row[0]:>6}  {row[1]:>6}  {row[2]:>5}")
        report_lines.append("")

        # ROC AUC
        report_lines.append("-" * 70)
        report_lines.append("ROC AUC SCORES")
        report_lines.append("-" * 70)
        report_lines.append("")
        for class_name, data in roc_data.items():
            report_lines.append(f"  {class_name.upper()}: {data['auc']:.4f}")
        report_lines.append("")

        report_lines.append("=" * 70)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 70)

        report = "\n".join(report_lines)

        # Save report
        with open(save_path, 'w') as f:
            f.write(report)

        logger.info(f"Evaluation report saved to {save_path}")

        return report

    def run_full_evaluation(
        self,
        generate_plots: bool = True,
        show_plots: bool = False
    ) -> Dict:
        """
        Run complete evaluation pipeline.

        Args:
            generate_plots: Whether to generate visualization plots
            show_plots: Whether to display plots

        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Running full evaluation...")

        results = {
            'classification_metrics': self.compute_classification_metrics(),
            'confusion_matrix': self.compute_confusion_matrix().tolist(),
            'roc_curves': self.compute_roc_curves()
        }

        if generate_plots:
            logger.info("Generating plots...")

            self.plot_confusion_matrix(show_plot=show_plots)
            self.plot_roc_curves(show_plot=show_plots)

            # Try to plot training history if available
            history_paths = [
                "logs/training_history.json",
                "../logs/training_history.json"
            ]
            for history_path in history_paths:
                if os.path.exists(history_path):
                    self.plot_training_history(history_path, show_plot=show_plots)
                    break

        # Generate report
        report = self.generate_report()

        logger.info("Full evaluation complete!")
        logger.info(f"Results saved to {self.output_dir}")

        return results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate action classification model")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                        help="Directory containing test data")
    parser.add_argument("--output-dir", type=str, default="results/evaluation",
                        help="Directory to save evaluation results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating visualization plots")
    parser.add_argument("--show-plots", action="store_true",
                        help="Display plots instead of just saving them")
    parser.add_argument("--counting", action="store_true",
                        help="Evaluate counting accuracy (requires ground truth counts)")

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )

    # Run evaluation
    results = evaluator.run_full_evaluation(
        generate_plots=not args.no_plots,
        show_plots=args.show_plots
    )

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {results['classification_metrics']['overall']['accuracy']:.4f}")
    print(f"F1 Score (macro): {results['classification_metrics']['overall']['f1_macro']:.4f}")
    print(f"F1 Score (weighted): {results['classification_metrics']['overall']['f1_weighted']:.4f}")
    print("=" * 50 + "\n")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
