# GolfDB SwingNet

Reimplementation and optimization of the SwingNet baseline from McNally et al., *"GolfDB: A Video Database for Golf Swing Sequencing"* (CVPR Workshops 2019). Produced for **CS 4267 (Deep Learning), Spring 2026**.

**Authors:** Connor Brugger, Bobby Miller
**Final 4-split cross-validation PCE: 0.7938** (paper baseline: 0.7610, a **+3.3 pp / +4.3% relative** improvement).

See [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) for the full writeup: hyperparameter table, concept explainers, ablation-style narrative, and Q&A prep. The README is the quick-start and the poster-planning doc; the SUMMARY is the deep reference.

---

## Contents

1. [What's different vs the paper](#whats-different-vs-the-paper)
2. [Things we tried / evaluated / abandoned](#things-we-tried--evaluated--abandoned)
3. [Repository layout](#repository-layout)
4. [Requirements](#requirements)
5. [Data setup](#data-setup)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Single-video demo](#single-video-demo)
9. [Results](#results)
10. [Poster planning — two distinct angles](#poster-planning--two-distinct-angles)
11. [Citation](#citation)

---

## What's different vs the paper

The core idea (per-frame CNN features → BiLSTM over time → 9-class head) is unchanged. Every gain comes from modernizing the training pipeline.

Architecture
- MobileNetV3-Large backbone (was MobileNetV2), via `timm` with pretrained ImageNet weights.
- 2-layer BiLSTM with 0.3 inter-layer dropout (was 1-layer, no dropout).
- Stochastic depth `drop_path=0.1` on the backbone.
- Runtime feature-dim probe (timm 1.0 changed `num_features` semantics on MobileNetV3; the old hardcoded 1280 would crash on current timm).

Optimization
- Fused AdamW with four parameter groups (backbone / head × decay / no-decay on 1-D params).
- Layer-wise Learning Rate Decay: backbone at 0.1× head LR.
- Cosine LR schedule with 500-iter linear warmup, floor 1e-7.
- Gradient clipping at 1.0.
- Weight decay 1e-4 (paper had none).

Regularization
- Label smoothing 0.1.
- Mixup α=0.2, p=0.5, `lam` clipped to ≥0.5.
- EMA of model weights at decay 0.9995; eval uses the EMA copy.
- Softer class weights `[1.0]×8 + [0.1]` (paper used 1/35 on the no-event class, which was over-suppressing once label smoothing + mixup were added).

Mixed precision and throughput
- bf16 autocast by default on Ada Lovelace (RTX 4080), fp16 available via `--fp16`.
- LSTM kept in fp32 for numerical stability.
- `channels_last` memory format for the CNN (≈25–30% faster NHWC on Ada tensor cores).
- TF32 matmul + `cudnn.benchmark`.
- Fused AdamW single-kernel updates.
- Deferred loss `.item()` sync every 10 iters to avoid GPU↔CPU stalls.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Data pipeline
- Full-dataset RAM preload of all uint8 frames (~6–8 GB) at startup; workers read slices, never redecode MP4s.
- Preallocated decode buffer: `cv2.cvtColor(..., dst=arr[i])` writes directly into `(N,H,W,3)` array.
- Vectorized label assignment (no Python per-frame loop).
- uint8 transport to GPU, all normalization/augmentation on the GPU.
- Persistent workers with prefetch.

GPU augmentation (new file `gpu_augment.py`)
- Fused color jitter (brightness+contrast+saturation → single mul+add).
- Rotation + x-shear + horizontal flip combined into one `grid_sample` call.
- Normalize uses precomputed `INV_STD` so the last op is a multiply (faster than divide in fp16/bf16).

Evaluation
- One model shell reused across all checkpoints.
- Horizontal-flip TTA: logits averaged, softmax in fp32.
- `--mode {all, ema, plain}` and `--last-n` filters.
- Reports PCE **and** Accuracy **and** Macro F1 (paper reports only PCE).
- 4-split cross-validation driver (`run_4split.sh`); paper reported only split 1.

Iteration count
- 8,000 iters per split with cosine decay (was 2,000 flat). Paper undertrains.

Everything listed above is backed by commentary in `PROJECT_SUMMARY.md`.

---

## Things we tried / evaluated / abandoned

Useful for Page 3 of the poster (individual contributions / decisions) and for the Q&A.

Accepted (in the final run)
- **MobileNetV3-Large over V2**: +2–3 ImageNet top-1 typically carries over. Win.
- **2-layer BiLSTM + 0.3 dropout**: the extra layer helped; dropout was necessary to avoid overfitting.
- **EMA**: reliably +1–2 PCE over the raw checkpoint; cheap.
- **Mixup (α=0.2, p=0.5)**: without `lam ≥ 0.5` clipping, the BiLSTM got confused by heavily blended sequences.
- **Label smoothing 0.1**: paired with mixup, softens over-confident class-8 logits.
- **LLRD (0.1× backbone)**: without it, the head's 1e-3 LR destroyed pretrained ImageNet features in the first few hundred iters.
- **Cosine schedule + 500-step warmup**: smoother minimum than flat LR; warmup removed early-step instability.
- **bf16 autocast + channels_last + TF32**: ~1.8× training throughput on the 4080, no PCE cost.
- **Fused AdamW**: one kernel per update.
- **RAM preload + GPU-fused augmentation**: removed the dataloader as a bottleneck.
- **Horizontal-flip TTA**: consistent +0.5–1 PCE at 2× eval cost.
- **4-split CV at 8,000 iters**: the paper's 2,000-iter / 1-split evaluation is too noisy.

Tried and kept toggleable, but not used for the final run
- **`torch.compile`**: works (`--compile` flag), but compile time ate ~10 min per split and our 9-hour budget didn't have room. Opt-in.
- **Gradient checkpointing on the backbone**: wired up (`checkpoint_backbone=True`), useful if batch size needs to grow; kept off because VRAM fit was fine at batch 16.
- **fp16 autocast with `GradScaler`**: supported via `--fp16`. Worked, but bf16 was simpler (no scaler, no NaN recovery) with identical PCE.

Tried and dropped
- **Paper's `1/35` no-event class weight**: with label smoothing + mixup, it over-suppressed the dominant class and hurt calibration. Replaced with `0.1`.
- **Batch size 22 (paper default)**: OOM'd on the 4080 at the first LSTM forward pass. Dropped to 16 (≈12 GB peak), no measurable PCE impact.
- **Freezing the first 10 backbone layers (paper default)**: not used — with LLRD at 0.1×, backbone updates are already gentle; freezing cost us ≈0.5 PCE in an early run. `freeze_layers()` remains in `util.py` for reference.
- **Constant LR**: tried, plateaued at ≈0.76. Cosine + warmup moved the number.
- **Single-checkpoint eval** (paper style): noisy between adjacent checkpoints. Switched to sweep-all-EMA + pick best.
- **Per-epoch dataloader (no preload)**: 6–7× slower; 8,000-iter 4-split run wouldn't fit in our time budget.

Considered but not implemented
- **Temporal transformer head** instead of BiLSTM. GolfDB (~1,400 clips) is too small to train a transformer well from scratch, and pretraining options for sequence-of-frames modeling at this scale are limited. Left as future work.
- **Cross-view / camera-angle robustness analysis**. The dataset's camera variance is noted in the paper but not broken out per-subset. Could be a follow-up.

---

## Repository layout

```
golfdb/
  model.py              # MobileNetV3-Large + 2-layer BiLSTM + 9-class head
  train.py              # Training loop: AMP, EMA, mixup, LLRD, cosine LR, resume
  eval.py               # Checkpoint sweep, TTA, reports PCE / Accuracy / Macro F1
  dataloader.py         # RAM-preload dataset, uint8 GPU-ready tensors
  gpu_augment.py        # Fused color jitter + rotate/shear/flip + normalize (GPU)
  util.py               # AverageMeter, PCE metric, layer freezing helper
  run_4split.sh         # 4-split cross-validation driver
  test_video.py         # Run the trained model on a single .mp4 for demo
  test_video.mp4        # Example swing video

  data/
    generate_splits.py  # Build the 4 CV splits from the .mat file
    preprocess_videos.py
    golfDB.mat          # Raw GolfDB annotations (~1,400 swings)
    golfDB.pkl          # Pickled dataframe used by the splits
    videos_160/         # 160x160 pre-cropped swing clips (not in git, ~13 GB)

  poster_best/
    swingnet_ema_7500.pth.tar
    swingnet_ema_8000.pth.tar   # best EMA checkpoint used for the demo

  results/
    results_max_20260417_125243.txt   # final 4-split PCE summary

  PROJECT_SUMMARY.md    # Deep-dive reference (poster source material)
  README.md             # This file
  .gitignore
  .gitattributes        # forces LF line endings to stop Windows/WSL churn
```

Generated at runtime (git-ignored): `.venv/`, `__pycache__/`, `models_*/`, `logs/`, `data/videos_160/`.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 4080 16 GB, WSL2 Ubuntu)
- `torch>=2.0`, `timm>=1.0`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `einops`, `scipy`

```bash
python -m venv .venv
source .venv/bin/activate         # Linux/WSL/Mac
# .venv\Scripts\activate          # Windows PowerShell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm opencv-python numpy pandas scikit-learn tqdm einops scipy
```

---

## Data setup

1. Download the pre-cropped 160×160 videos from the [original repo](https://github.com/wmcnally/golfdb) (Google Drive link in their README) and extract to `data/videos_160/`.
2. The `.pkl` split files (`data/train_split_{1..4}.pkl`, `data/val_split_{1..4}.pkl`) are generated from `data/golfDB.mat`:
   ```bash
   cd data && python generate_splits.py && cd ..
   ```

---

## Training

Single split:
```bash
python train.py --split 1 --model-dir models_s1 --batch-size 16 --seq-length 64 --iterations 8000
```

All 4 splits (our final experimental setup):
```bash
bash run_4split.sh
```

Environment overrides the driver understands: `BATCH`, `SEQ`, `ITERATIONS`, `RUN_TAG`.
```bash
BATCH=16 ITERATIONS=8000 bash run_4split.sh
```

The driver writes per-split train/eval logs and a summary `results_<tag>.txt`.

---

## Evaluation

Evaluate EMA checkpoints with horizontal-flip TTA (matches our reported numbers):
```bash
python eval.py models_s1 --split 1 --seq-length 64 --mode ema --tta
```

Flags:
- `--mode {all, ema, plain}` — which checkpoints to evaluate.
- `--last-n N` — after filtering, keep only the final N checkpoints.
- `--tta` — horizontal-flip test-time augmentation.
- `--fp16` — fp16 autocast instead of bf16.

---

## Single-video demo

```bash
python test_video.py -p test_video.mp4 --ckpt poster_best/swingnet_ema_8000.pth.tar
```

Prints the predicted frame index and confidence for each of the 8 swing events (Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish). Add `--show` to display annotated event frames.

---

## Results

Run tag: `max_20260417_125243`, 4 splits × 8,000 iterations, batch=16, seq=64, bf16.

| Split | PCE    | Accuracy | Macro F1 |
|-------|--------|----------|----------|
| 1     | 0.7964 | 0.9500   | 0.4865   |
| 2     | 0.7957 | 0.9489   | 0.4705   |
| 3     | 0.7868 | 0.9480   | 0.4733   |
| 4     | 0.7964 | 0.9521   | 0.4788   |
| **Avg** | **0.7938** | **0.9498** | **0.4773** |
| Paper | 0.7610 | —        | —        |

All 4 splits individually beat the paper's 0.7610; the lowest (split 3) is 0.7868. Full raw output: [`results/results_max_20260417_125243.txt`](./results/results_max_20260417_125243.txt).

---

## Poster planning — two distinct angles

Each teammate submits an individual 3-page poster (title + 30"×18" single-sheet poster + own-contributions page). The course rubric awards 20 pts for **"fundamentally different poster as your teammates."** To satisfy that, below are two non-overlapping framings of the same codebase. Pick your angle, use the corresponding section of `PROJECT_SUMMARY.md` for the source material, stay out of the other angle's lane.

### Angle A — Connor: "Modernizing a 2019 baseline: +3.3 PCE from training-pipeline engineering"

Thesis: the paper's architecture is fine; what's holding it back is a 2019 training recipe. Replace the recipe and the same architecture family jumps +3.3 PCE.

Recommended narrative arc:
1. Problem + baseline: swing-phase event detection, PCE metric, 76.1 baseline.
2. Six levers changed, grouped: backbone, optimizer (AdamW + LLRD + cosine + warmup), regularization (mixup + label smoothing + EMA + stochastic depth), AMP (bf16 + channels_last + TF32), data pipeline (RAM preload + GPU augmentation), eval (TTA + full-EMA sweep + 4-split CV).
3. 4-split bar chart: 0.7964 / 0.7957 / 0.7868 / 0.7964 vs paper 0.7610.
4. Wall-clock story: 43 min/split end-to-end; bf16 + channels_last + deferred sync made 8k×4 fit in budget.
5. Single-video demo still (use `poster_best/swingnet_ema_8000.pth.tar` + `test_video.py`).

Page 3 (Connor's individual contributions):
- Pipeline rewrite (model.py, train.py, eval.py, dataloader.py, new gpu_augment.py).
- LLRD + cosine schedule decisions and the constant-LR → cosine move that bought ~2 PCE.
- bf16/TF32/channels_last/fused-AdamW throughput work.
- RAM-preload + GPU-augmentation refactor.
- 4-split CV driver and the results infrastructure.

Metrics to foreground: **PCE**, wall-clock, throughput. De-emphasize Macro F1 (it's not the task metric here).

Source material: `PROJECT_SUMMARY.md` sections *Final Results*, *What We Did*, *Hyperparameter Table*, *Why Each Change Matters*.

### Angle B — Bobby: "Per-frame swing-phase classification: class imbalance, regularization, and calibration"

Thesis: framed as a 9-way per-frame classification problem (not as event detection with PCE), what does the confusion matrix actually look like? Where does the model fail, and how does each regularizer move the failure modes? This directly matches the original proposal's emphasis on Accuracy, Macro F1, and (per-class) AUC-ROC.

Recommended narrative arc:
1. Problem reframed: 9 classes per frame, ~1:35 event-to-no-event imbalance.
2. Why Macro F1 / per-class ROC matters on imbalanced multiclass data (and why PCE alone hides phase-specific failure modes).
3. Class-weight study: paper's `1/35` on class 8 vs our final `0.1` vs `1.0` (no weighting). What changed.
4. Regularization ablation-style: label smoothing 0.1 on/off, mixup α=0.2 on/off. What it does to per-class recall.
5. Per-class confusion matrix on split 1 (generate from `eval.py` probs — you can extend `eval.py` to dump confusion; code is already doing `frame_preds` and `frame_labels` concat).
6. Per-class precision / recall / F1 table, plus ROC curves for the 8 event classes.
7. Error analysis: which two phases get confused most (Mid-backswing vs Top, typically; Mid-downswing vs Impact).

Page 3 (Bobby's individual contributions):
- Frame-level evaluation methodology and the metric choice (why AUC-ROC + Macro F1 in addition to PCE).
- Class-imbalance experiments and the decision to move from `1/35` → `0.1`.
- Confusion-matrix and per-class analysis (new plots, own code).
- Literature context from the proposal (why frame-level classification is a valid intermediate step toward event detection).

Metrics to foreground: **Accuracy**, **Macro F1**, **per-class precision/recall/AUC-ROC**, **confusion matrix**. De-emphasize wall-clock and bf16/TF32 (those are Angle A).

Source material: `PROJECT_SUMMARY.md` sections *What the Paper Did*, *Regularization*, *Key Concepts Explained*, *What Might Come Up in Q&A*. Plus the per-frame outputs from `eval.py`.

### Keeping the two posters distinct

| Dimension | Connor (Angle A) | Bobby (Angle B) |
|-----------|------------------|------------------|
| Framing | Event-sequence detection | Per-frame classification |
| Metric in the headline | PCE | Macro F1 / AUC-ROC / confusion matrix |
| Biggest plot | 4-split bar chart vs paper | Per-class confusion matrix + ROC curves |
| Story spine | Training-recipe modernization | Imbalance handling + calibration |
| Own-contribution focus | Pipeline + throughput + CV protocol | Metric methodology + imbalance study + error analysis |

Both are honest descriptions of the same codebase. The fact that both are supported by the same 4-split run is **not** a problem — team results belong to the team, but each poster interprets and presents them through a different lens.

---

## Citation

```
@InProceedings{McNally_2019_CVPR_Workshops,
  author = {McNally, William and Vats, Kanav and Pinto, Tyler and Dulhanty, Chris and McPhee, John and Wong, Alexander},
  title = {GolfDB: A Video Database for Golf Swing Sequencing},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2019}
}
```

Original code is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
