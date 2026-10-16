#!/usr/bin/env bash
# One-shot setup on macOS: venv, dependencies, prior map, tests, one headless mission.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN=python3.12
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN=python3.11
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN=python3
else
  echo "ERROR: Please install Python 3.11 or 3.12, then rerun." >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit("ERROR: Python >= 3.11 required; install python3.12 using Homebrew or python.org")
print("Using Python", sys.version.split()[0])
PY
"$PYTHON_BIN" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
python -m robotcode.cli --dry-run --task "Đi vào phòng khách, tìm chai nước trên bàn, lấy nó rồi mang đến cho tao."
python -m robotcode.cli --smoke --headless
python -m robotcode.cli --headless --quiet --scenario deliver_bottle_to_user
printf '\nREADY. Open the 3D simulator with:\n  source .venv/bin/activate\n  mjpython -m robotcode.cli --interactive\n'
