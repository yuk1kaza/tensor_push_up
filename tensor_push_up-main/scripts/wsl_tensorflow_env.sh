#!/usr/bin/env bash

activate_tensor_push_up_wsl_env() {
  local venv_dir="${1:-.venv-wsl}"

  if [[ ! -d "${venv_dir}" ]]; then
    echo "Virtual environment not found: ${venv_dir}" >&2
    return 1
  fi

  # shellcheck disable=SC1090
  source "${venv_dir}/bin/activate"

  local detected_libs
  detected_libs="$(python - <<'PY'
import pathlib
import site

paths = []
for base in site.getsitepackages():
    nvidia_root = pathlib.Path(base) / "nvidia"
    if nvidia_root.exists():
        paths.extend(str(path) for path in sorted(nvidia_root.glob("*/lib")) if path.is_dir())

print(":".join(paths))
PY
)"

  if [[ -n "${detected_libs}" ]]; then
    export LD_LIBRARY_PATH="${detected_libs}:${LD_LIBRARY_PATH:-}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  activate_tensor_push_up_wsl_env "${1:-.venv-wsl}"
fi
