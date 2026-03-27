# Labels Guide

This folder stores JSON label files used by `src/preprocess.py`.

## Current Status

- `pushup_dataset_labels.json` was auto-generated from the current contents of `data/raw/`
- `jumping_jack_dataset_labels.json` was auto-generated after adding `jumping_jack*.mp4`
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
