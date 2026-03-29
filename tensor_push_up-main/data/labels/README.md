# Labels Guide

This folder stores JSON label files used by `src/preprocess.py`.

## Current Status

- `pushup_dataset_labels.json` was auto-generated from the current contents of `data/raw/`
- `jumping_jack_dataset_labels.json` was auto-generated after adding `jumping_jack*.mp4`
- `other_dataset_labels.json` is ready for future `other*.mp4` videos
- the current dataset now contains both `pushup` and `jumping_jack`
- `count` is left as `null` and should be filled in manually if you know the real repetition count

## Current Limitation

The dataset is no longer single-class, so training can proceed. However, it still
does not contain an `other` class. That means:

- `pushup` vs `jumping_jack` classification is now meaningful
- but the model still cannot learn a true three-class setup until you add
  `other` videos and labels

## Next Recommended Steps

1. Review `pushup_dataset_labels.json` and `jumping_jack_dataset_labels.json`
2. Fill in the real `count` values if you have them
3. If you want a full three-class model, add `other` videos and labels
4. Re-run:

```bash
python src/preprocess.py --input data/raw --output data/processed
```

5. Then run:

```bash
bash scripts/train_wsl.sh --venv .venv-wsl --smoke
```

## Folder-First Conventions

The project now includes a helper script that generates label files using the
video folder first, then the filename as a fallback:

```bash
python scripts/generate_labels_from_filenames.py --input data/raw --labels data/labels
```

Recommended dataset layout:

```text
data/raw/
  pushup/
    *.mp4
  jumping_jack/
    *.mp4
  other/
    *.mp4
```

Folder mapping:

- `pushup/` or `push_up/` -> `pushup`
- `jumping_jack/` -> `jumping_jack`
- `other/` -> `other`

Fallback filename prefixes:

- `push_up*.mp4` -> `pushup`
- `jumping_jack*.mp4` -> `jumping_jack`
- `other*.mp4` -> `other`

If a video has no matching label entry yet, preprocessing now tries to infer its
class from the folder first and will auto-generate the missing JSON entry in the
corresponding label file.
