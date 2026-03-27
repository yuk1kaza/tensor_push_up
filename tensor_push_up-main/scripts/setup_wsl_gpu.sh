#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv-wsl"
PYTHON_BIN="python3"
REQUIREMENTS_FILE="requirements.txt"
SKIP_APT=0
INDEX_URL="${PIP_INDEX_URL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_FILE="$2"
      shift 2
      ;;
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --index-url)
      INDEX_URL="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: scripts/setup_wsl_gpu.sh [options]

Options:
  --venv PATH            Virtual environment directory. Default: .venv-wsl
  --python BIN           Python binary to use. Default: python3
  --requirements PATH    requirements.txt path. Default: requirements.txt
  --skip-apt             Skip apt package installation
  --index-url URL        Override pip index URL, e.g. https://pypi.tuna.tsinghua.edu.cn/simple
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

PIP_INSTALL_ARGS=()
if [[ -n "${INDEX_URL}" ]]; then
  PIP_INSTALL_ARGS+=(-i "${INDEX_URL}")
  echo "[setup] Using pip index: ${INDEX_URL}"
fi

if [[ "${SKIP_APT}" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
  echo "[setup] Installing Ubuntu prerequisites..."
  sudo apt-get update
  sudo apt-get install -y \
    python3-venv python3-pip build-essential git grep \
    libgl1 libegl1 libgles2 libopengl0
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[setup] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo "[setup] Upgrading pip tooling..."
python -m pip install "${PIP_INSTALL_ARGS[@]}" --upgrade pip setuptools wheel

echo "[setup] Installing TensorFlow with CUDA extras..."
python -m pip install "${PIP_INSTALL_ARGS[@]}" "tensorflow[and-cuda]"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "requirements file not found: ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

FILTERED_REQUIREMENTS="$(mktemp)"
grep -v '^[[:space:]]*tensorflow' "${REQUIREMENTS_FILE}" > "${FILTERED_REQUIREMENTS}" || true

if [[ -s "${FILTERED_REQUIREMENTS}" ]]; then
  echo "[setup] Installing remaining project dependencies..."
  python -m pip install "${PIP_INSTALL_ARGS[@]}" -r "${FILTERED_REQUIREMENTS}"
else
  echo "[setup] No non-TensorFlow dependencies found to install."
fi

rm -f "${FILTERED_REQUIREMENTS}"

cat <<EOF

[done] WSL GPU environment setup complete.

Next steps:
  1. bash scripts/verify_wsl_gpu.sh --venv ${VENV_DIR}
  2. bash scripts/train_wsl.sh --venv ${VENV_DIR} --smoke
  3. bash scripts/train_wsl.sh --venv ${VENV_DIR}
EOF
