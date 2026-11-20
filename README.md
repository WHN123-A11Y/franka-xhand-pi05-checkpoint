# Franka + XHand π₀.₅ SFT — step 30000

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
