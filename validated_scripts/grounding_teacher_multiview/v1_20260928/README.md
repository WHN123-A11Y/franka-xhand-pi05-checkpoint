# 三视角 Teacher 固定版 v1_20260928

已验证范围：RoboTwin-Clean / adjust_bottle / episode 8 / frame 0 / 一个对象 / 三视角离线冒烟测试。不是完整定位基准、Student 验证或机器人动作成功率验证。

## 保存内容

- `smoke.py`：按真实 task_index 找 episode，核对 parquet，解码三视角，运行 Teacher。
- `export_results.py`：导出每视角 Top-3、跨视角最终最佳框及评分。
- `source/`：七个 Teacher 文件及两个候选池模块的固定副本；最小包入口避免依赖上传目录中无关的模块。
- `fixtures/language_results.jsonl`：本例已验证的 v4.1 语言解析输出，运行时不再依赖历史 runs 中的语言文件，也不重新调用语言模型。
- `fixtures/validated_case.json`、`PROVENANCE.json`、`SHA256.json`：验证配置、来源及内容哈希。

固定脚本只修改了工程根目录、Teacher 来源和默认语言输入路径；候选生成、每视角筛选和跨视角 BT 算法保持原样。运行时仍在新 runs 目录生成路径适配快照。

## 环境与外部依赖

工程根：`/data/adminad/workspace/FineVLA_clean`

Python：`/data/adminad/miniforge3/envs/finevla/bin/python`

版本：torch 2.6.0+cu124、transformers 5.2.0、av 12.3.0、pyarrow 14.0.1、Pillow 12.3.0。GPU 验证环境为 A100 80GB；使用 PyAV/libdav1d 解码 AV1，SDPA 推理。

模型：`/data/adminad/models/Qwen3.5-4B`。数据：`/data/adminad/datasets/RoboTwin-Clean/adjust_bottle`。模型与视频不复制到脚本目录。

源码基础 Git revision：`29983a5494bf52cf88fd944ebf9ce857bc9fa4c4`。实验源码和上传源码包含未跟踪文件，因此实际版本以 `PROVENANCE.json` 和 `SHA256.json` 为准，不能只依靠 Git revision。

## 测试命令

```bash
cd /data/adminad/workspace/FineVLA_clean
CUDA_VISIBLE_DEVICES=1 HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=8 \
/data/adminad/miniforge3/envs/finevla/bin/python -u \
validated_scripts/grounding_teacher_multiview/v1_20260928/smoke.py \
--dataset RoboTwin-Clean --task adjust_bottle --task-index 8 --frame 0
```

约 5 分钟。每次生成新的 `runs/grounding_teacher_smoke_<时间>/`。导出时用终端打印的实际 Output 目录：

```bash
/data/adminad/miniforge3/envs/finevla/bin/python \
validated_scripts/grounding_teacher_multiview/v1_20260928/export_results.py \
--run /data/adminad/workspace/FineVLA_clean/runs/grounding_teacher_smoke_实际时间
```

其他样本需显式传入包含对应原始指令的 `--language-results`；只有本例已经通过端到端冒烟验证。

## 验证记录

- 初次成功：`runs/grounding_teacher_smoke_20260928_085035_820838/`，约 294.97 秒。
- 用户复测成功：`runs/grounding_teacher_smoke_20260928_085918_402834/`，约 296.04 秒，GROUP=PASS、BEST VIEW=cam_high。
- 发布检查：固定版根路径和所有 Teacher 依赖导入通过；没有加载 GPU 模型重跑。
- 固定版导出器在临时目录读取用户复测的结果，生成的 `best_candidate.png`、`three_view_top3.png`、`summary.json` 与已确认产物 SHA256 完全一致。

主视角框覆盖瓶子，但边界偏宽；两个腕部视角看不到目标，仍输出候选，跨视角评分降低了其权重。评分不是可见概率或同一物体的关联保证。

## 后续维护

此目录作为固定版本保存。后续修改在 `experiments/grounding_teacher_multiview/` 开发，验证后发布新版本，不在这里调试。产物继续放 runs，不写入本目录。
