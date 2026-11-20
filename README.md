# Franka + XHand π₀.₅ SFT — step 30000

这是 π₀.₅ 在 Franka + XHand 数据集上完成 30000 步 SFT 后的**完整 checkpoint**，包括模型参数和续训状态。已有 Student、K4 配套的是按验证集选出的 step 1500 基线，不能未经重新验证就将此 step 30000 与现有 K4 直接组合使用。当前没有真机闭环成功结果。

在 Release 中下载全部 `pi05_xhand_sft_30000.tar.zst.part-*` 文件到 `parts/`，同时将 `SHA256SUMS.txt`、`CHECKPOINT_FILES.sha256` 和 `extract.sh` 下载到 `parts/` 的上一级；在那里执行 `bash extract.sh`。脚本会先核验分卷，再解压 `30000/` 并核验每个 checkpoint 文件。

仓库为私有，下载者需要先获得仓库访问权限。若已经登录 GitHub CLI，可在空目录中使用：

```bash
gh release download sft-step-30000 --repo WHN123-A11Y/franka-xhand-pi05-checkpoint --pattern 'pi05_xhand_sft_30000.tar.zst.part-*' --dir parts
gh release download sft-step-30000 --repo WHN123-A11Y/franka-xhand-pi05-checkpoint --pattern 'SHA256SUMS.txt' --pattern 'CHECKPOINT_FILES.sha256' --pattern 'extract.sh' --dir .
bash extract.sh
```

This artifact contains the complete recorded `30000/` checkpoint from the Franka + XHand fine-tuning run: model parameters, normalization assets and training state for resuming. It is separate from the **step 1500** checkpoint selected for the existing Student and K4 training. Do not assume that the existing K4 is validated with this checkpoint.

The full SFT ran for 30000 steps. On the recorded validation metric, step 1500 flow MSE was 0.0760497; step 30000 was 0.1759443. This offline metric is not robot task success; neither checkpoint has been shown here to succeed in real-robot closed-loop evaluation.

## Download and verify

Download every `pi05_xhand_sft_30000.tar.zst.part-*` release asset into a directory called `parts/`, and put `SHA256SUMS.txt`, `CHECKPOINT_FILES.sha256` and `extract.sh` in its parent. Use the SHA-256 list to check the downloaded parts, then run:

```bash
sha256sum -c SHA256SUMS.txt
bash extract.sh
```

The script requires Linux/macOS shell tools, `zstd`, `tar` and `sha256sum`; it refuses to overwrite an existing `30000/` directory and checks checkpoint-file hashes after extraction. Keep about 61 GiB free for the parts plus extracted checkpoint. The original source directory was a completed local checkpoint; no robot actions are performed by extraction.

Training setup: `pi05_xhand`, dataset `franka_xhand_pi05_pick_place`, batch size 8, fixed validation seed 1701, 19 supervised action dimensions and 10-step action horizon. Recorded source revision `215abfb217dbac7d5f1273282331b9b1866c0479` had local changes; a clean checkout is not guaranteed equivalent. See the separate project handoff for the actual captured native OpenPI source, environment, dataset details and hardware integration limitations.

The complete checkpoint is about 30.1 GiB uncompressed; standalone `params/` is approximately 11.6 GiB and `train_state/` approximately 18.5 GiB. `SHA256SUMS.txt` verifies archive parts, while `CHECKPOINT_FILES.sha256` verifies their extracted contents.
