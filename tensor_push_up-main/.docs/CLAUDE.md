# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tensor Push Up is a computer vision system that counts push-ups and jumping jacks from video streams (webcam or video files). It uses MediaPipe for pose estimation and TensorFlow for action classification with state machine-based counting.

## Architecture

The system follows a two-stage pipeline:

### 1. Pose Estimation (`src/pose_estimator.py`)
- Uses MediaPipe to extract 33 body keypoints
- Calculates 8 joint angles for motion analysis
- Outputs normalized features (50-dimensional vectors per frame)

### 2. Action Classification & Counting
- **Model**: LSTM with 2 layers (128, 64 units) and dropout
- **Input**: Sliding window of 30 frames (temporal sequence)
- **Counter**: State machines with stability frames and cooldown periods
- **Output**: 3-class classification (pushup, jumping_jack, other)

## Key Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Pipeline
```bash
# Preprocess raw videos to training data
python src/preprocess.py --input data/raw --output data/processed

# View data statistics
python src/preprocess.py --stats --input data/processed
```

### Training
```bash
# Train model with default config
python src/train.py --config configs/train.yaml

# Train with custom parameters
python src/train.py --config configs/train.yaml --epochs 50 --batch_size 64
```

### Evaluation
```bash
# Evaluate model on test set
python src/evaluate.py --model models/checkpoints/best.keras --data data/processed

# Generate evaluation report
python src/evaluate.py --model models/checkpoints/best.keras --data data/processed --report
```

### Inference
```bash
# Real-time webcam counting
python src/infer.py --source 0 --model models/checkpoints/best.keras --task count

# Process video file
python src/infer.py --source demo.mp4 --model models/checkpoints/best.keras --task count

# Save annotated output video
python src/infer.py --source demo.mp4 --model models/checkpoints/best.keras --task count --save-video
```

### Demo
```bash
# Quick demo of pose estimation
python demo.py --mode pose --source 0

# Quick demo of action counting
python demo.py --mode count --source 0
```

## Configuration

All training parameters are in `configs/train.yaml`:
- Model architecture (LSTM units, dropout)
- Data parameters (window size, splits, batch size)
- Augmentation settings
- Action counting thresholds
- Confidence thresholds

The system uses a configuration-driven approach - modify the YAML file to experiment with different settings without code changes.

## Counting Logic

The system uses state machines for stable counting:

### Push-up Counter
- High position: elbow angle > 150°
- Low position: elbow angle < 90°
- Count when: high → low → high sequence detected
- Stability: 3 frames minimum, 10 frames cooldown

### Jumping Jack Counter
- Open position: arms up, feet apart
- Closed position: arms down, feet together
- Count when: closed → open → closed sequence detected
- Uses normalized distances between keypoints

## Security Measures

The project includes security validation:
- File path validation to prevent directory traversal
- File size limits (500MB for videos, 100MB for models)
- Extension validation for all input files
- Input sanitization for filenames

Security functions are in `src/security.py` and used by:
- `VideoReader` and `VideoWriter` classes
- Model loading functions
- Configuration loading

## Testing

No formal test suite is implemented. Testing is done through:
1. Demo scripts (`demo.py`)
2. Manual inference testing
3. Evaluation script (`evaluate.py`)

## Data Format

Processed data consists of:
- `X_train.npy`, `X_val.npy`, `X_test.npy`: Feature arrays (samples, 30, 50)
- `y_train.npy`, `y_val.npy`, `y_test.npy`: Label arrays (0=pushup, 1=jumping_jack, 2=other)

Features are normalized joint angles calculated from pose keypoints.