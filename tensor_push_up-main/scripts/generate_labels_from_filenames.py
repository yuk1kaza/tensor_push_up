"""
Generate label JSON files from directory names first, then filename prefixes.

Supported naming conventions:
- data/raw/pushup/*.mp4 -> pushup
- data/raw/push_up/*.mp4 -> pushup
- data/raw/jumping_jack/*.mp4 -> jumping_jack
- data/raw/other/*.mp4 -> other

Fallback filename prefixes:
- push_up*.mp4 -> pushup
- jumping_jack*.mp4 -> jumping_jack
- other*.mp4 -> other

This is a convenience script for bootstrapping label files when raw videos
follow predictable names.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import cv2


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

ACTION_DIR_ALIASES = {
    "pushup": {"pushup", "push_up"},
    "jumping_jack": {"jumping_jack", "jumpingjack"},
    "other": {"other"},
}

ACTION_TO_FILENAME = {
    "pushup": "pushup_dataset_labels.json",
    "jumping_jack": "jumping_jack_dataset_labels.json",
    "other": "other_dataset_labels.json",
}


def detect_action_type(video_path: Path) -> Optional[str]:
    for part in reversed(video_path.parts[:-1]):
        lowered = part.lower()
        for action, aliases in ACTION_DIR_ALIASES.items():
            if lowered in aliases:
                return action

    lowered = video_path.name.lower()
    if lowered.startswith("push_up") or lowered.startswith("pushup"):
        return "pushup"
    if lowered.startswith("jumping_jack") or lowered.startswith("jumpingjack"):
        return "jumping_jack"
    if lowered.startswith("other"):
        return "other"
    return None


def get_frame_count(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()


def build_label_record(video_path: Path, action_type: str) -> Dict:
    frame_count = get_frame_count(video_path)
    return {
        "action_type": action_type,
        "count": None,
        "start_frame": 0,
        "end_frame": max(frame_count - 1, 0),
        "notes": (
            "Auto-generated from filename. Fill in the actual repetition count "
            "if available."
        ),
    }


def load_existing_labels(label_file: Path) -> Dict:
    if not label_file.exists():
        return {}
    return json.loads(label_file.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate label files from raw video filenames")
    parser.add_argument("--input", default="data/raw", help="Directory containing raw videos")
    parser.add_argument("--labels", default="data/labels", help="Directory to write label JSON files")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing entries instead of only adding missing videos",
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    labels_dir = Path(args.labels)
    labels_dir.mkdir(parents=True, exist_ok=True)

    grouped_updates: Dict[str, Dict[str, Dict]] = {
        "pushup": {},
        "jumping_jack": {},
        "other": {},
    }

    for video_path in sorted(input_dir.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTS:
            continue

        action_type = detect_action_type(video_path)
        if action_type is None:
            continue

        grouped_updates[action_type][video_path.name] = build_label_record(video_path, action_type)

    for action_type, updates in grouped_updates.items():
        label_filename = ACTION_TO_FILENAME[action_type]
        label_file = labels_dir / label_filename
        existing = load_existing_labels(label_file)

        if args.overwrite:
            merged = {**existing, **updates}
        else:
            merged = dict(existing)
            for video_name, record in updates.items():
                merged.setdefault(video_name, record)

        if not merged:
            continue

        label_file.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"{label_file}: {len(merged)} entries")


if __name__ == "__main__":
    main()
