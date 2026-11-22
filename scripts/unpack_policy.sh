#!/usr/bin/env bash
# Verify the downloaded parts, then extract the selected step-1500 policy.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
command -v zstd >/dev/null || { echo '缺少 zstd，请先安装。' >&2; exit 1; }
command -v sha256sum >/dev/null || { echo '缺少 sha256sum。' >&2; exit 1; }
[[ -f selected_files_sha256.txt ]] || { echo '缺少根目录校验清单，请下载交付目录里的两个 sha256 清单。' >&2; exit 1; }
[[ ! -e models/pi05_selected_1500 ]] || { echo 'models/pi05_selected_1500 已存在；为避免覆盖，请先自行检查或移走。' >&2; exit 1; }
if [[ -f weights/policy_parts_github.sha256 ]]; then
  sha256sum -c weights/policy_parts_github.sha256
  PARTS=(weights/pi05_selected_1500.tar.zst.part-{00..06})
elif [[ -f policy_parts_github.sha256 ]]; then
  sha256sum -c policy_parts_github.sha256
  PARTS=(weights/pi05_selected_1500.tar.zst.part-{00..06})
else
  sha256sum -c policy_parts_sha256.txt
  PARTS=(weights/pi05_selected_1500.tar.zst.part-{00..02})
fi
mkdir -p models
STAGING_DIR="$(mktemp -d "models/.pi05_unpack.XXXXXXXX")"
trap 'rm -rf -- "$STAGING_DIR"' EXIT
cat "${PARTS[@]}" | zstd -dc | tar -xf - -C "$STAGING_DIR"
[[ -d "$STAGING_DIR/pi05_selected_1500" ]] || { echo '归档内找不到策略目录。' >&2; exit 1; }
(cd "$STAGING_DIR/pi05_selected_1500" && sha256sum -c "$PROJECT_DIR/selected_files_sha256.txt")
mv -- "$STAGING_DIR/pi05_selected_1500" models/pi05_selected_1500
echo '完成。模型已校验并解压到 models/pi05_selected_1500/。'
