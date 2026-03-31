"""
Download a Kaggle dataset into the project and print its file structure.

Example:
    python scripts/download_kaggle_dataset.py \
        --dataset mrigaankjaswal/exercise-detection-dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path


def list_files(base_dir: Path, limit: int = 200) -> None:
    files = sorted(p for p in base_dir.rglob("*") if p.is_file())
    print(f"Downloaded file count: {len(files)}")
    for path in files[:limit]:
        print(path.relative_to(base_dir))
    if len(files) > limit:
        print(f"... and {len(files) - limit} more")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and inspect a Kaggle dataset")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Kaggle dataset handle, e.g. mrigaankjaswal/exercise-detection-dataset",
    )
    parser.add_argument(
        "--output-dir",
        default="data/external",
        help="Directory where the dataset should be downloaded",
    )
    args = parser.parse_args()

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "kagglehub is not installed. Run: pip install kagglehub"
        ) from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset {args.dataset} into {output_dir} ...")
    dataset_path = Path(
        kagglehub.dataset_download(
            args.dataset,
            force_download=False,
            output_dir=str(output_dir),
        )
    )

    print(f"Dataset downloaded to: {dataset_path}")
    list_files(dataset_path)


if __name__ == "__main__":
    main()
