#!/usr/bin/env python3
"""Check the files used by the local offline replay; does not start a robot."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
EXPECTED = {
    "models/student/best.pt": "6b6cae24798833bcd41d0056f13f44d4c8fb354b6719ea3c0700c1a709b80b48",
    "models/student/language.pt": "b06fc7aaff7520dfef570d59cddc32acbbe83fd5ecdd7e88ff9f265520d2e83b",
    "models/k4/best.pt": "715f8c4ba86abdb2c7a5d9b282c98380d43f4da91df67d99ae02fdba5756a678",
    "models/pi05_selected_1500/assets/xhand_local/norm_stats.json": "4341bda15d6abe1184b3c2d0060b64a0c598fbd76b13161ec2a81ddd957dc24a",
}
REQUIRED = (
    "src/offline_inference.py", "src/common.py", "src/student_model.py",
    "native_openpi/src/openpi/training/config.py",
    "native_openpi/packages/openpi-client/src/openpi_client/__init__.py",
    "models/student/expressions.json", "assets/openpi_cache/big_vision/paligemma_tokenizer.model",
    "samples/ep000002_f000000/input_sample.npz", "samples/ep000002_f000000/reference.npz",
    "models/pi05_selected_1500/params/_METADATA",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-policy-hash", action="store_true", help="校验已解压 π₀.₅ 的全部 19 个文件，需读取约 12 GiB")
    parser.add_argument("--weights-only", action="store_true", help="仅检查源码和权重；不要求私下提供的真实记录帧")
    args = parser.parse_args()
    errors = []
    for relative in REQUIRED:
        if args.weights_only and relative.startswith("samples/"):
            continue
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"缺少 {relative}")
    for relative, expected in EXPECTED.items():
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"缺少 {relative}")
            continue
        if digest(path) != expected:
            errors.append(f"SHA-256 不符 {relative}")
        else:
            print(f"OK  {relative}")
    if args.full_policy_hash:
        manifest = ROOT / "selected_files_sha256.txt"
        if not manifest.is_file():
            errors.append("缺少 selected_files_sha256.txt")
        else:
            for line in manifest.read_text().splitlines():
                expected, name = line.split("  ", 1)
                name = name.removeprefix("./")
                path = ROOT / "models/pi05_selected_1500" / name
                if not path.is_file() or digest(path) != expected:
                    errors.append(f"策略文件缺失或 SHA-256 不符: {name}")
            if not errors:
                print("OK  策略权重 19 个文件的 SHA-256")
    if errors:
        for e in errors:
            print(f"ERROR  {e}", file=sys.stderr)
        print("检查工程目录或先运行 scripts/unpack_policy.sh；详见 README。", file=sys.stderr)
        return 1
    print("检查通过。仅说明所选文件检查通过；未执行推理或真机控制。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
