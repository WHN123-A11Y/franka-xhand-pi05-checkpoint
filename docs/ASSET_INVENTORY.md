# 配套源码、模型与外部资源清单

公开配套资产：[student-k4-step1500](https://github.com/WHN123-A11Y/franka-xhand-pi05-checkpoint/releases/tag/student-k4-step1500)。所有下载资产的固定大小和 SHA-256 在 [paired_release_manifest.json](../assets/paired_release_manifest.json)。

| 内容 | 工程路径 | 来源与版本 |
|---|---|---|
| π₀.₅ 动作基线 | `models/pi05_selected_1500/` | 训练总30000步，验证选择1500；7分卷恢复，全部19文件校验 |
| 动作归一化 | `models/pi05_selected_1500/assets/xhand_local/norm_stats.json` | 随1500基线归档，无需另找 |
| Student 权重 | `models/student/best.pt` | step2000；定位和候选验证头 |
| Student 配套 | `models/student/{language.pt,expressions.json,feature_config.json,PARTIAL_READY.json}` | 同一训练版本，ZIP提供 |
| K4 权重 | `models/k4/best.pt` | step12273；基于1500基线训练 |
| K4 记录 | `models/k4/{READY.json,summary.json}` | ZIP提供；原服务器路径仅作来源记录 |
| Tokenizer | `assets/openpi_cache/big_vision/paligemma_tokenizer.model` | ZIP提供，附 `licenses/LICENSE_GEMMA.txt` |
| 离线核心代码 | `src/{offline_inference.py,common.py,student_model.py}` | Git源码 |
| π₀.₅ 实现及训练入口 | `native_openpi/src/`、`native_openpi/scripts/train.py` | 实际使用的修改版，源码来源记录随仓库提供 |
| Student/K4 训练 | `experiments/real_franka_student_four_gpu_v1/`、`experiments/real_franka_k4_four_gpu_v1/` | Git源码；迁移训练仍需数据与本机配置 |
| Teacher 标注 | `experiments/real_franka_teacher_all_v1/`、`experiments/refcoco_eval/teacher_adapter.py` 及快照依赖 | Git源码；Teacher模型另行提供 |
| 真实离线样本 | `samples/ep000002_f000000/{input_sample.npz,reference.npz}` | 团队私下交接，未公开 |
| 原始真机数据/Teacher模型 | 依原训练配置 | 未公开，不是在线Student/K4推理权重 |

Student 的 `view_validity` 和 `view_distribution` 没有训练，本版使用已训练的候选验证分数融合。不能把文件完整理解成这些头已训练。

`step 30000` 是独立权重快照，位于另一 Release；本配套 Student/K4 未验证在30000基线上运行。

2026-11-22 检查：7分卷远端digest与本地逐字节校验一致；归档19/19文件一致；实际下载ZIP校验通过；干净克隆中执行恢复脚本和权重完整性检查。大分卷恢复测试使用已验证且与远端digest一致的缓存，没有重复下载约12GB策略。完整新机环境安装、训练迁移和真机闭环未在本次检查中执行。
