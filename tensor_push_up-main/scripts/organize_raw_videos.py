"""
Organize flat raw videos into class subdirectories.

Rules:
1. Existing JSON labels take priority.
2. If no label exists, infer class from parent folder or filename prefix.
3. Move videos into:
   - data/raw/pushup/
   - data/raw/jumping_jack/
   - data/raw/other/
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, Optional


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ACTION_DIR_ALIASES = {
    "pushup": {"pushup", "push_up"},
    "jumping_jack": {"jumping_jack", "jumpingjack"},
    "other": {"other"},
}


def load_labels(labels_dir: Path) -> Dict[str, Dict]:
    labels: Dict[str, Dict] = {}
    if not labels_dir.exists():
        return labels

    for json_file in sorted(labels_dir.glob("*.json")):
        try:
            labels.update(json.loads(json_file.read_text(encoding="utf-8")))
        except Exception:
            continue
    return labels


def infer_action_type(video_path: Path, labels: Dict[str, Dict]) -> Optional[str]:
    labeled = labels.get(video_path.name, {})
    action = labeled.get("action_type")
    if action:
        return action

    for part in reversed(video_path.parts[:-1]):
        lowered = part.lower()
        for action_type, aliases in ACTION_DIR_ALIASES.items():
            if lowered in aliases:
                return action_type

    lowered = video_path.name.lower()
    if lowered.startswith("push_up") or lowered.startswith("pushup"):
        return "pushup"
    if lowered.startswith("jumping_jack") or lowered.startswith("jumpingjack"):
        return "jumping_jack"
    if lowered.startswith("other"):
        return "other"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Organize raw videos into class folders")
    parser.add_argument("--input", default="data/raw", help="Raw video directory")
    parser.add_argument("--labels", default="data/labels", help="Labels directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned moves without changing files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    labels_dir = Path(args.labels)
    labels = load_labels(labels_dir)

    moved = 0
    skipped = 0

    for video_path in sorted(input_dir.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        action_type = infer_action_type(video_path, labels)
        if action_type is None:
            print(f"SKIP {video_path} -> unknown class")
            skipped += 1
            continue

        target_dir = input_dir / action_type
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / video_path.name

        if video_path.resolve() == target_path.resolve():
            continue

        if target_path.exists():
            print(f"SKIP {video_path} -> {target_path} already exists")
            skipped += 1
            continue

        print(f"MOVE {video_path} -> {target_path}")
        if not args.dry_run:
            shutil.move(str(video_path), str(target_path))
        moved += 1

    print(f"Done. moved={moved}, skipped={skipped}")


if __name__ == "__main__":
    main()
