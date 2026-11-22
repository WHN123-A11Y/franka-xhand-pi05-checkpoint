# Franka + XHand π₀.₅ / Teacher / Student / K4

本仓库保留本次真机数据实验使用的核心推理源码、训练参考源码和 Teacher 依赖快照。权重通过下方公开 Release 下载；原始数据、真实相机样本、运行日志和缓存不放入 Git 历史。

## 代码目录

- `src/`：已在记录帧上验证的离线预测入口（`offline_inference.py`）、Student 模型、K4 条件融合和数据接口。不连接或控制真机。
- `native_openpi/`：π₀.₅ 运行所需的实际修改版 OpenPI `src/` 和 `openpi-client` 快照，保留原许可证。源码的5个原空标记文件仅增加换行以兼容文件传输。上游 Git revision 为 `215abfb217dbac7d5f1273282331b9b1866c0479`，训练时工作树含修改，不能用干净上游版本保证等价。
- `experiments/real_franka_sft_eight_gpu_v1/` 与 `experiments/real_franka_full_training_v1/policy/`：原始 SFT、数据准备及数据加载关键脚本。
- `experiments/real_franka_student_four_gpu_v1/`、`experiments/pi05_native_student_20261016/model.py`：Student 特征、训练目标和模型；`experiments/real_franka_full_training_v1/student/prepare.py` 为历史前置数据准备代码。
- `experiments/real_franka_k4_four_gpu_v1/`：K4 训练及条件融合实现。
- `experiments/real_franka_teacher_all_v1/`、`experiments/refcoco_eval/teacher_adapter.py`：真机数据 Teacher 工作进程、类别颜色协议和适配器。适配器动态使用 `validated_scripts/grounding_teacher_multiview/v1_20260928/` 冻结源码和 `runs/finevla_50_shared_20261005_052510/teacher/runtime/vendor/` 中的投影、DINO 和核验代码；它们是**源码快照**，不是运行结果。
- `scripts/`：环境创建、checkpoint 解包和离线运行辅助脚本。`.vscode/` 与 `Franka-XHand.code-workspace` 可在 VS Code 中使用。

历史训练和 Teacher 脚本保留原服务器绝对路径、运行目录约定与原有数据/模型依赖；公开 Release 已补齐离线推理所需配套权重，但对应训练数据和 Teacher 模型另行提供，**迁移到新机器前须适配，尚未在新机器端到端验证**。请勿将其视为即开即用的真机控制器，也不要通过填入伪造数据或随机权重绕过校验。已取消的 207 条 Teacher 补标任务并未继续。

## 模型与运行

本次 SFT 训练 30000 步，按验证集选用 step **1500**；Student 选 step **2000**；K4 选 step **12273**。Student 只训练定位和候选框验证，view validity / view distribution 未训练；K4 使用候选分数归一化融合。区域均无效时 K4 旁路。Student/K4 在 step 1500 基线上训练及评估，**未验证替换成 step 30000 的组合**。

配套 **step 1500 + Student + K4 + tokenizer + 归一化参数** 已在[公开配套 Release](https://github.com/WHN123-A11Y/franka-xhand-pi05-checkpoint/releases/tag/student-k4-step1500) 中（7 个策略分卷 + `paired_assets.zip` + 校验清单）。独立的 [step 30000 Release](https://github.com/WHN123-A11Y/franka-xhand-pi05-checkpoint/releases/tag/sft-step-30000) 保留，不能直接替代 K4 的训练基线。下载和恢复命令见下文；无需 GitHub 登录。真实记录帧和训练数据仍由团队私下交接。依赖以 `environment/requirements_native.txt` 记录；环境安装需 Python 3.11、网络和兼容 GPU/CUDA，尚未在新机器验证。输入物理相机为 head、front-left、wrist；模型槽中第二路的 `left_wrist` 是历史命名，并非物理腕部相机。动作是记录口径的 7 维机械臂 + 12 维灵巧手关节，动作头为 10×32，仅前19维有监督。真实控制器关节顺序、单位、限位、时序、时钟和安全策略均需设备负责人核对。

有团队授权的使用者可从百度网盘 `franka_xhand_vscode_20261120/` 下载 `工程源码与配置.zip`（含 Student、K4 和样本）及 `weights/` 中 step 1500 分卷。准备好资源后，在仓库根目录运行 `bash scripts/unpack_policy.sh`、`bash scripts/setup_native_env.sh` 和 `bash scripts/run_offline.sh --gpu 0`。新机器的环境安装仍需验证；这不是硬件控制入口。

完整离线测试集 flow MSE：baseline 0.0640287，K4 0.0637174，下降约 0.49%；不是成功率。独立离线入口真实帧上同次 native/zero 输出一致，跨进程有小幅数值波动且原因未查明。**未进行本版本真机闭环评估，不能据此宣称抓取成功或可部署。**

## 下载、恢复和检查

在 Linux/WSL 中准备 `git`、`curl`、`python3`、`zstd` 和 `tar`，然后运行：

```bash
git clone https://github.com/WHN123-A11Y/franka-xhand-pi05-checkpoint.git
cd franka-xhand-pi05-checkpoint
bash scripts/fetch_paired_assets.sh
python3 scripts/check_assets.py --weights-only --full-policy-hash
```

下载器固定 Release 版本和每个文件的字节数/SHA-256，支持断点下载；解压到临时目录后，逐一检查19个策略文件和 Student/K4/tokenizer 的原始哈希，再恢复到工程路径。已有模型目录会拒绝覆盖；下载中断后可重跑。`weights/` 缓存可在校验和恢复后自行删除以节约空间。[完整资产清单](docs/ASSET_INVENTORY.md)。

运行真实记录帧仍需要团队交接包里的 `samples/ep000002_f000000/input_sample.npz` 和 `reference.npz`；GitHub 不含真实相机图片。这两个文件放好后，再执行：

```bash
bash scripts/setup_native_env.sh
.venv/bin/python scripts/check_assets.py --full-policy-hash
bash scripts/run_offline.sh --gpu 0
```

下载和校验通过只说明文件配套。离线入口曾在原服务器真实记录帧验证；新机器依赖安装和实际推理仍需本机验证。本次完整性检查没有重跑八卡训练或真机控制。

历史训练源码已补 `native_openpi/scripts/train.py`。重新训练还需要原始数据、Teacher 权重和修正训练配置中的服务器路径；K4 对历史 READY/feature_config 中的绝对模型路径有严格校验，迁移时须保留原件并生成经过哈希校验的本机配置副本，不能仅复制源码就称一键训练。

## Teacher 边界

Teacher 为 Qwen3.5-4B + Grounding DINO 的离线标注流程，生成的是伪标签；它与 Student/K4 在线推理是分开的。历史 Teacher 仅核验颜色、物体类别，不保证形状或方位正确。冻结代码中的部分文件名带 `test_`，但作为 TeacherAdapter 动态加载的运行依赖保留；没有复制实验运行数据。当前 Teacher 参考源码需要单独提供模型、数据、配置并处理硬编码路径，未经新环境验证，不能直接运行。

详细工程和真机交接资料位于团队百度网盘 `franka_xhand_vscode_20261120/`；该交付包含权重与数据，不等于本公开源码仓库的内容。
