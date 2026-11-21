#!/usr/bin/env bash
# Replay a recorded sample: no controller connection, no hardware command.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
GPU="${FRANKA_OFFLINE_GPU:-0}"
PYTHON_BIN="${FRANKA_OFFLINE_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
OUT="${FRANKA_OFFLINE_OUT:-$PROJECT_DIR/runs/offline_replay}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    -h|--help) echo '用法: bash scripts/run_offline.sh [--gpu 物理卡号] [--python Python路径] [--out 结果目录]'; exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done
[[ -x "$PYTHON_BIN" ]] || { echo "找不到可执行 Python: $PYTHON_BIN；请创建 .venv 或用 --python 指定。" >&2; exit 1; }
# K4 was trained with the validation-selected step-1500 base. Check all
# 19 model files so a checkpoint with identical normalization but different
# weights (e.g. the step-30000 snapshot) cannot be silently substituted.
"$PYTHON_BIN" scripts/check_assets.py --full-policy-hash
echo '正在离线重放记录帧；不连接或控制真机。'
exec "$PYTHON_BIN" src/offline_inference.py \
  --native-source "$PROJECT_DIR/native_openpi" \
  --checkpoint "$PROJECT_DIR/models/pi05_selected_1500" \
  --student "$PROJECT_DIR/models/student/best.pt" \
  --language "$PROJECT_DIR/models/student/language.pt" \
  --expressions "$PROJECT_DIR/models/student/expressions.json" \
  --k4 "$PROJECT_DIR/models/k4/best.pt" \
  --tokenizer-cache "$PROJECT_DIR/assets/openpi_cache" \
  --sample "$PROJECT_DIR/samples/ep000002_f000000/input_sample.npz" \
  --reference "$PROJECT_DIR/samples/ep000002_f000000/reference.npz" \
  --gpu "$GPU" --out "$OUT"
