# Ubuntu WSL2 GPU Training Guide

This project can run on native Windows, but TensorFlow `>= 2.11` commonly falls
back to CPU-only there. For NVIDIA GPU training, the recommended setup is:

`Windows NVIDIA driver -> WSL2 -> Ubuntu -> Python venv -> TensorFlow GPU in Linux`

This guide is written for the current project path:

```text
D:\Programs\VScode\tensor_push_up-main\tensor_push_up-main
```

Inside Ubuntu WSL, the same path becomes:

```bash
/mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
```

## What You Need

- Windows 11 or a recent Windows 10 build with WSL2 support
- An NVIDIA GPU
- The latest Windows-side NVIDIA driver that supports CUDA on WSL
- Ubuntu installed through WSL2

## Important Rule

Do not install a separate Linux NVIDIA display driver inside WSL. The Windows
driver is the driver stack that WSL GPU compute relies on.

## 1. Prepare Windows

Open PowerShell and check that the GPU is visible on Windows:

```powershell
nvidia-smi
```

If this fails, fix the Windows driver first before touching WSL.

Install or update WSL:

```powershell
wsl --install -d Ubuntu
wsl --update
wsl -l -v
```

Make sure your Ubuntu distro shows version `2`.

If you already have Ubuntu installed but it is still WSL1:

```powershell
wsl --set-version Ubuntu 2
```

## 2. Open Ubuntu WSL

Launch Ubuntu, then update base packages:

```bash
sudo apt update
sudo apt upgrade -y
```

Optional but useful tools:

```bash
sudo apt install -y python3-venv python3-pip build-essential git
```

## 3. Enter The Project

```bash
cd /mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
pwd
```

You should now be inside the project from the Linux side.

## 4. Create A Linux Virtual Environment

Do not reuse the Windows venv from `.venv\Scripts\activate`. Create a Linux-side
venv from Ubuntu:

```bash
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python --version
```

If you use VS Code, the cleanest workflow is opening the repo through
`Remote - WSL` and selecting `.venv-wsl` as the interpreter.

## 5. Install TensorFlow With GPU Support In WSL

Upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install TensorFlow with bundled CUDA user-space dependencies:

```bash
python -m pip install "tensorflow[and-cuda]"
```

If the download speed is extremely low, do not just wait indefinitely. A
hundreds-of-megabytes TensorFlow wheel downloading at only a few kB/s usually
means your connection to the default PyPI index is the bottleneck. In that case,
cancel and retry with a closer mirror.

Example with the Tsinghua mirror:

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "tensorflow[and-cuda]"
```

The project setup script also supports this directly:

```bash
bash scripts/setup_wsl_gpu.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Then install the rest of the project dependencies.

Because `requirements.txt` also contains `tensorflow>=2.12.0`, the safest WSL
flow is to install the non-TensorFlow packages separately:

```bash
grep -v "^tensorflow" requirements.txt > /tmp/tensor_push_up_requirements_wsl.txt
python -m pip install -r /tmp/tensor_push_up_requirements_wsl.txt
```

If `grep` is unavailable for some reason, create the filtered file manually and
exclude the TensorFlow line.

## 6. Verify GPU Visibility Inside WSL

Check whether WSL can access the GPU:

```bash
nvidia-smi
```

If `nvidia-smi` is not on `PATH`, try:

```bash
/usr/lib/wsl/lib/nvidia-smi
```

Then verify TensorFlow itself:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Expected result:

- At least one `PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')`
- No fallback to an empty list `[]`

## 7. Verify The Project Data Before Training

This project now fails fast when the dataset is unusable. A common example is
having only one class such as `other`.

Check whether you already have processed data:

```bash
ls data/processed
```

If you still need to preprocess:

```bash
python src/preprocess.py --input data/raw --output data/processed
```

If you have not created label JSON files under `data/labels`, preprocessing will
default most samples to `other`, and training will stop early with a warning.

## 8. Start GPU Training From Ubuntu WSL

Normal training:

```bash
python src/train.py --config configs/train.yaml
```

Short smoke test:

```bash
python src/train.py --config configs/train.yaml --epochs 1 --no-export
```

If you deliberately want to test the pipeline on placeholder single-class data:

```bash
python src/train.py --config configs/train.yaml --epochs 1 --no-export --allow-single-class
```

## 9. What The Training Logs Mean Now

The trainer now logs:

- current platform
- TensorFlow version
- visible GPU devices
- actual dataset input shape
- warnings when config shape does not match processed data
- warnings when classes are missing

Typical healthy WSL startup logs should show:

- `platform=linux`
- at least one visible GPU device

Typical native Windows logs will show:

- `platform=win32`
- `gpus=[]`
- a warning telling you to move training into WSL2

## 10. Recommended VS Code Workflow

From Windows:

1. Install the `WSL` extension in VS Code.
2. Run `WSL: Connect to WSL`.
3. Open `/mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main`.
4. Select `.venv-wsl/bin/python` as the interpreter.
5. Run training from the WSL terminal inside VS Code.

This avoids mixing Windows Python, Windows TensorFlow, and Linux TensorFlow.

## 11. Common Problems

### `tf.config.list_physical_devices('GPU')` returns `[]`

Check in this order:

1. `nvidia-smi` works on Windows.
2. `wsl -l -v` shows Ubuntu is on version `2`.
3. `nvidia-smi` works inside Ubuntu WSL.
4. You installed `tensorflow[and-cuda]` inside the Linux venv, not in Windows PowerShell.
5. You are actually running `python` from `.venv-wsl`.

### Training says only one class is present

That is a data problem, not a GPU problem.

Check:

- `data/labels/` contains actual JSON label files
- preprocessing was re-run after labels were added
- `data/processed/train_labels.npy` is no longer all `2`

### The model input shape in config does not match processed data

The trainer now warns and uses the real dataset shape, but you should still
regenerate data or update config so the repo stays consistent.

### I already installed CUDA Toolkit on Windows

That is fine, but for this project the main point is that WSL GPU compute must
work end-to-end. The Windows driver and WSL2 integration matter more than
manually reproducing a full Linux CUDA stack inside the distro.

## 12. Minimal Copy-Paste Path

If your Windows driver and WSL2 are already healthy, this is the shortest path:

```bash
cd /mnt/d/Programs/VScode/tensor_push_up-main/tensor_push_up-main
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel
bash scripts/setup_wsl_gpu.sh --index-url https://pypi.tuna.tsinghua.edu.cn/simple
grep -v "^tensorflow" requirements.txt > /tmp/tensor_push_up_requirements_wsl.txt
python -m pip install -r /tmp/tensor_push_up_requirements_wsl.txt
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python src/train.py --config configs/train.yaml
```

## References

- TensorFlow pip install guide: <https://www.tensorflow.org/install/pip>
- Microsoft WSL install guide: <https://learn.microsoft.com/windows/wsl/install>
- Microsoft GPU compute in WSL overview: <https://learn.microsoft.com/windows/ai/directml/gpu-cuda-in-wsl>
- NVIDIA CUDA on WSL user guide: <https://docs.nvidia.com/cuda/wsl-user-guide/index.html>
