#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-wsl"
CONFIG_PATH="configs/train.yaml"
DATA_DIR=""
SMOKE=0
ALLOW_SINGLE_CLASS=0
VERIFY_FIRST=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --data-dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --allow-single-class)
      ALLOW_SINGLE_CLASS=1
      shift
      ;;
    --skip-verify)
      VERIFY_FIRST=0
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/train_wsl.sh [options]

Options:
  --venv PATH               Virtual environment directory. Default: .venv-wsl
  --config PATH             Training config path. Default: configs/train.yaml
  --data-dir PATH           Override processed data directory
  --smoke                   Run a one-epoch smoke test with no export
  --allow-single-class      Allow smoke tests on single-class datasets
  --skip-verify             Skip GPU verification before training
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

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found: ${VENV_DIR}" >&2
  echo "Run bash scripts/setup_wsl_gpu.sh first." >&2
  exit 1
fi

source scripts/wsl_tensorflow_env.sh
activate_tensor_push_up_wsl_env "${VENV_DIR}"

if [[ "${VERIFY_FIRST}" -eq 1 ]]; then
  bash scripts/verify_wsl_gpu.sh --venv "${VENV_DIR}"
fi

CMD=(python src/train.py --config "${CONFIG_PATH}")

if [[ -n "${DATA_DIR}" ]]; then
  CMD+=(--data-dir "${DATA_DIR}")
fi

if [[ "${SMOKE}" -eq 1 ]]; then
  CMD+=(--epochs 1 --no-export)
fi

if [[ "${ALLOW_SINGLE_CLASS}" -eq 1 ]]; then
  CMD+=(--allow-single-class)
fi

echo "[train] ${CMD[*]}"
"${CMD[@]}"
