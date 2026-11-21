#!/usr/bin/env bash
# Offline-model Python environment only; does not install Teacher or control hardware.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_PYTHON="${FRANKA_PYTHON311:-python3.11}"
VENV_DIR="$PROJECT_DIR/.venv"
REQ_FILE="$PROJECT_DIR/environment/requirements_native.txt"
NATIVE_DIR="$PROJECT_DIR/native_openpi"

if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
  echo "错误：$VENV_DIR 已存在，不会覆盖。请检查现有环境；确需重建时由您自行处理后重新运行。" >&2
  exit 1
fi
command -v "$BASE_PYTHON" >/dev/null || { echo "找不到 Python 3.11：$BASE_PYTHON。可设置 FRANKA_PYTHON311 指向可执行文件。" >&2; exit 1; }
"$BASE_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), f"需要 Python 3.11，当前为 {sys.version}"'
[[ -f "$REQ_FILE" && -f "$NATIVE_DIR/pyproject.toml" && -f "$NATIVE_DIR/packages/openpi-client/pyproject.toml" ]] || {
  echo '缺少 environment/requirements_native.txt 或 native_openpi 源码；请检查工程是否完整。' >&2; exit 1;
}

echo '创建项目专用 .venv，依赖会从网络下载；新机器和 GPU 环境尚未端到端验证。'
echo '若安装中断，.venv 会保留供检查；脚本不会覆盖已有环境。'
"$BASE_PYTHON" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m ensurepip
# The historical capture pins the actually used dependency versions. Avoid a
# new dependency resolution against metadata from another OpenPI revision.
"$VENV_DIR/bin/python" -m pip install --no-deps -r "$REQ_FILE"
"$VENV_DIR/bin/python" -m pip install 'hatchling==1.27.0' 'editables==0.5'
"$VENV_DIR/bin/python" -m pip install --no-deps --no-build-isolation -e "$NATIVE_DIR/packages/openpi-client" -e "$NATIVE_DIR"
echo "完成：$VENV_DIR/bin/python。请继续执行离线检查；此脚本未安装 Teacher，也未测试或控制真机。"
