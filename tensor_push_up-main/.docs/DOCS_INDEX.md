# Docs Index

This folder contains project guides, GPU training notes, inference usage, and dataset requirements.

## Core Docs

- [Project Overview](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/README.md)
- [Development Guide](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/DEVELOPMENT.md)
- [API Reference](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/API.md)
- [Claude Notes](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/CLAUDE.md)

## WSL2 GPU Training

- [Ubuntu WSL2 GPU Training Guide](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/WSL_GPU_TRAINING.md)
- [Ubuntu WSL2 GPU 训练指南](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/WSL_GPU_TRAINING_CN.md)
- [Ubuntu WSL2 GPU 训练执行计划](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/WSL_GPU_TRAINING_PLAN_CN.md)

## Dataset And Inference

- [Quick Start](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/QUICK_START.md)
- [数据集与标注要求](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/Data_Requirement.md)
- [项目运行流程图](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/PROJECT_RUNTIME_FLOW_CN.md)
- [训练后模型计数使用说明](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/INFERENCE_USAGE_CN.md)
- [特征工程落地计划](d:/Programs/VScode/tensor_push_up-main/tensor_push_up-main/.docs/FEATURE_ENGINEERING_PLAN_CN.md)

## Current Highlights

- Three classes are currently supported in the dataset: `pushup`, `jumping_jack`, `other`
- The MediaPipe feature extraction bug that produced all-zero features has been fixed
- The trained model can now be used for offline video counting
- Push-up and jumping-jack counters have been tuned against the current dataset
- `demo.py` and `infer.py` now have documented usage differences for webcam vs. video input
