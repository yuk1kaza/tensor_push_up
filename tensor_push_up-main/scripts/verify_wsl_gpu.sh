#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-wsl"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/verify_wsl_gpu.sh [--venv PATH]
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "src/train.py" ]]; then
  echo "Please run this script from the project root." >&2
  exit 1
fi

source scripts/wsl_tensorflow_env.sh
activate_tensor_push_up_wsl_env "${VENV_DIR}"

echo "[verify] Host platform:"
uname -a

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[verify] nvidia-smi"
  nvidia-smi
elif [[ -x "/usr/lib/wsl/lib/nvidia-smi" ]]; then
  echo "[verify] /usr/lib/wsl/lib/nvidia-smi"
  /usr/lib/wsl/lib/nvidia-smi
else
  echo "[verify] nvidia-smi not found inside WSL." >&2
fi

echo "[verify] TensorFlow GPU visibility"
python - <<'PY'
import platform
import sys

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"platform={sys.platform}")
print(f"release={platform.release()}")
print(f"tensorflow={tf.__version__}")
print(f"gpus={gpus}")

if not gpus:
    raise SystemExit("No GPU devices visible to TensorFlow.")
PY

echo "[done] TensorFlow can see at least one GPU."
