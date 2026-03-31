"""
Package trained model artifacts for GitHub Release uploads.

This script copies the main model artifacts and metadata into a release folder
and creates a zip archive for the SavedModel export.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def main() -> None:
    parser = argparse.ArgumentParser(description="Package model artifacts for GitHub Release")
    parser.add_argument("--name", default=None, help="Release asset base name")
    parser.add_argument(
        "--output-dir",
        default="release_assets",
        help="Directory where packaged release artifacts should be written",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d")
    release_name = args.name or f"tensor-push-up-model-{timestamp}"

    repo_root = Path(".").resolve()
    output_dir = (repo_root / args.output_dir / release_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    best_model = repo_root / "models" / "checkpoints" / "best_model.keras"
    h5_model = repo_root / "models" / "exported" / "action_classifier.h5"
    saved_model_dir = repo_root / "models" / "exported" / "action_classifier"
    metrics_file = repo_root / "logs" / "test_metrics.json"
    dataset_info_file = repo_root / "logs" / "dataset_info.json"

    required_paths = [best_model, h5_model, saved_model_dir, metrics_file, dataset_info_file]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required artifacts: {missing}")

    copied_best_model = output_dir / "best_model.keras"
    copied_h5_model = output_dir / "action_classifier.h5"
    copied_metrics = output_dir / "test_metrics.json"
    copied_dataset = output_dir / "dataset_info.json"

    shutil.copy2(best_model, copied_best_model)
    shutil.copy2(h5_model, copied_h5_model)
    shutil.copy2(metrics_file, copied_metrics)
    shutil.copy2(dataset_info_file, copied_dataset)

    saved_model_archive_base = output_dir / "action_classifier_savedmodel"
    archive_path = shutil.make_archive(
        str(saved_model_archive_base),
        "zip",
        root_dir=saved_model_dir.parent,
        base_dir=saved_model_dir.name,
    )

    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    dataset_info = json.loads(dataset_info_file.read_text(encoding="utf-8"))

    notes = f"""# {release_name}

## Included Assets

- `best_model.keras`
- `action_classifier.h5`
- `action_classifier_savedmodel.zip`
- `test_metrics.json`
- `dataset_info.json`

## Dataset Summary

- input shape: `{dataset_info.get('input_shape')}`
- train samples: `{dataset_info.get('train_samples')}`
- val samples: `{dataset_info.get('val_samples')}`
- test samples: `{dataset_info.get('test_samples')}`
- labels: `{dataset_info.get('available_labels')}`

## Evaluation Summary

- accuracy: `{metrics.get('accuracy'):.4f}`
- precision: `{metrics.get('precision'):.4f}`
- recall: `{metrics.get('recall'):.4f}`
- f1_score: `{metrics.get('f1_score'):.4f}`
- evaluated labels: `{metrics.get('evaluated_labels')}`

## File Sizes

- `best_model.keras`: {format_size(copied_best_model.stat().st_size)}
- `action_classifier.h5`: {format_size(copied_h5_model.stat().st_size)}
- `action_classifier_savedmodel.zip`: {format_size(Path(archive_path).stat().st_size)}
"""

    notes_path = output_dir / "RELEASE_NOTES.md"
    notes_path.write_text(notes, encoding="utf-8")

    print(f"Release assets prepared in: {output_dir}")
    for path in [copied_best_model, copied_h5_model, Path(archive_path), copied_metrics, copied_dataset, notes_path]:
        print(f"- {path.name} ({format_size(path.stat().st_size)})")


if __name__ == "__main__":
    main()
