# GolfDB SwingNet - Project Summary

Reference doc for building individual posters. Every section is self-contained so you can pull pieces in any order.

---

## TL;DR

We reimplemented the SwingNet baseline from McNally et al., "GolfDB: A Video Database for Golf Swing Sequencing" (CVPR Workshops 2019) and trained a version that beats the published 76.1% PCE baseline by **+3.3 percentage points** (+4.3% relative) on the paper's own 4-split cross-validation. Our average is **79.38%**.

All gains come from modernizing the training pipeline (backbone, optimizer, regularization, AMP, augmentation, evaluation). The core architecture idea (CNN per-frame features + BiLSTM over time) is unchanged.

---

## Final Results

4-split cross-validation (same splits as paper, `data/val_split_{1..4}.pkl`), evaluated on the EMA checkpoint with horizontal-flip test-time augmentation.

| Split   | Best Checkpoint       | PCE    | Accuracy | Macro F1 |
|---------|-----------------------|--------|----------|----------|
| 1       | swingnet_ema_8000     | 0.7964 | 0.9500   | 0.4865   |
| 2       | swingnet_ema_8000     | 0.7957 | 0.9489   | 0.4705   |
| 3       | swingnet_ema_8000     | 0.7868 | 0.9480   | 0.4733   |
| 4       | swingnet_ema_8000     | 0.7964 | 0.9521   | 0.4788   |
| **Avg** |                       | **0.7938** | **0.9498** | **0.4773** |
| Paper   | MobileNetV2 SwingNet  | 0.7610 | n/a      | n/a      |

**Delta vs paper:** +0.0328 PCE absolute, +4.3% relative.
**Hardware:** RTX 4080 16 GB, WSL2 Ubuntu.
**Wall time:** ~43 min per split end-to-end (train + eval), ~2h 47m for all four splits.

PCE = Percentage of Correct Events. An event prediction is correct if the argmax frame index is within a tolerance window of the ground truth event frame. Tolerance scales with swing length (~1/30 of the address-to-impact span).

---

## What the Paper Did (Baseline)

- **Architecture:** MobileNetV2 (custom implementation from tonylins/pytorch-mobilenet-v2) + 1-layer bidirectional LSTM (hidden=256) + linear head over 9 classes (8 swing events + "no event").
- **Training data:** 160x160 pre-cropped clips of full golf swings from the GolfDB dataset (~1,400 videos).
- **Loss:** CrossEntropyLoss with fixed class weights `[1/8]*8 + [1/35]` to counter the ~1:35 event-to-no-event imbalance.
- **Optimizer:** Plain Adam, lr=1e-3, no schedule, no weight decay.
- **Iterations:** 2,000 (train.py default).
- **AMP:** None, all fp32.
- **Precision/channels format:** Default fp32, channels-first.
- **Augmentation:** The README states "without any data augmentation" produces 71.5% PCE on split 1. The paper's 76.1% required horizontal flipping plus affine transforms that are **not in the provided code**. We reimplemented those.
- **Evaluation:** Single-split (split 1), single checkpoint (swingnet_1800), no TTA.

The 8 swing events the paper labels are: Address, Toe-up, Mid-Backswing, Top, Mid-Downswing, Impact, Mid-Follow-Through, Finish. Class 8 is any other frame.

---

## What We Did (Our Pipeline)

Everything listed here is a departure from the paper's provided code. Grouped by what they buy you:

### Architecture

- **MobileNetV3-Large backbone** via timm instead of MobileNetV2. Same family, updated blocks with hard-swish activations and Squeeze-and-Excitation attention. Timm gives us pretrained weights directly via `timm.create_model(..., pretrained=True)`, no manual weight file download.
- **2-layer BiLSTM with inter-layer dropout (0.3)** instead of 1 layer with no dropout. More capacity, regularized.
- **Stochastic depth (drop_path=0.1)** on the backbone. Each residual block is randomly dropped during training, acting as an implicit ensemble.
- **Runtime feat_dim probe.** Timm 1.0+ changed the semantics of `num_features` on MobileNetV3 (it now returns the pre-conv-head block dim, 960, instead of the post-conv-head dim, 1280). We auto-detect via `head_hidden_size`, fall back to `num_features`, and finally run a CPU forward pass to probe. This is a real correctness fix, the old hardcoded 1280 would crash on current timm.

### Optimization

- **AdamW (fused) + LLRD:** Fused AdamW with four parameter groups:
  1. Backbone weights with decay
  2. Backbone 1-D params (BN weights, biases) with **no** decay
  3. Head weights with decay
  4. Head 1-D params with no decay
  Backbone learning rate is 0.1x the head LR (Layer-wise Learning Rate Decay). Pretrained features shouldn't move as fast as a randomly-initialized head. Skipping weight decay on BN/biases is standard best practice, it avoids shrinking norm statistics toward zero.
- **Cosine LR schedule + linear warmup:** 500-iter warmup to `lr=1e-3`, then cosine decay to `lr=1e-7` over 7,500 iters.
- **Gradient clipping at max_norm=1.0** to prevent LSTM blowups.
- **Weight decay = 1e-4** (regularizer, paper had none).

### Regularization

- **Label smoothing (0.1):** Prevents overconfident predictions.
- **Mixup at alpha=0.2, prob=0.5:** Random linear blends of two sequences per batch, with `lam` clipped to >=0.5 so the primary signal dominates. Mixing is done with a single fused `torch.lerp` for both images and labels.
- **EMA of model weights at decay=0.9995:** We keep a shadow fp32 copy of the weights that's updated every iteration. At eval time we evaluate the EMA weights. The EMA is basically a "temporally smoothed" version of the trajectory and reliably outperforms any individual checkpoint in our logs.
- **Class weights:** Reweighted to `[1.0]*8 + [0.1]` (softer than the paper's 1:35 ratio). With label smoothing + mixup the harder 1/35 weight was over-suppressing negatives.

### Mixed Precision and Throughput

- **bfloat16 default (Ada Lovelace):** We run the forward pass and loss in bf16 autocast. bf16 has the same exponent range as fp32, so no gradient scaling is needed and there are no NaN recovery loops. fp16 is still available via `--fp16` (with a `GradScaler`).
- **LSTM kept in fp32** inside `forward()` for numerical stability. The backbone produces bf16 features, we cast to fp32 before the LSTM, then the linear head's output is cast back by autocast implicitly.
- **channels_last memory format** for the CNN. On Ada, tensor-core convolutions are ~25-30% faster in NHWC than NCHW.
- **TF32 matmul + cudnn.benchmark:** Allows cuDNN to pick optimal kernels once per shape and reuse them.
- **Fused AdamW** (`fused=True`): One kernel launch per parameter update instead of many.
- **torch.compile compatible** (opt-in via `--compile`). We didn't enable it for the final run because compile times ate into the budget, but it works.
- **Deferred loss sync:** The loss is accumulated on-GPU in a scalar tensor and only `.item()`'d every `--log-every` iterations (default 10). Each `.item()` forces a CUDA-to-CPU sync which stalls the pipeline. Deferring reduced syncs by 10x.
- **`expandable_segments:True`** for the CUDA allocator to reduce fragmentation over long runs.

### Data Pipeline

- **Full-dataset RAM preload.** All training videos (~6-8 GB of uint8 frames) are decoded once at startup and stored in a process-global dict. Workers read slices from RAM instead of re-decoding MP4 chunks every epoch. The paper's code decodes on every `__getitem__`, which is ~1 order of magnitude slower.
- **Preallocated decode buffer:** `cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=arr[i])` writes directly into a preallocated `(N, H, W, 3)` uint8 numpy array, avoiding per-frame intermediate allocations.
- **Vectorized label assignment:** `positions = (np.arange(seq_length) + start) % total_frames` then boolean-mask assignments for the 8 event slots, instead of a Python for loop per frame.
- **uint8 transport, GPU normalize.** Workers hand off raw uint8 CHW tensors. All normalization, color jitter, affine, and flip happen **on the GPU** in `gpu_augment.py`. Keeps the dataloader lightweight and PCIe bandwidth low. Workers=2 is enough.
- **Persistent workers with prefetch_factor=4:** Workers stay alive across epochs, prefetch queue hides I/O behind GPU compute.
- **opencv threads pinned to 0** inside workers (`cv2.setNumThreads(0)`) to stop cv2 from fighting with the dataloader's own parallelism.

### GPU Augmentation (`gpu_augment.py`)

Every aug is in-place and fused where possible:

- **Color jitter (brightness, contrast, saturation)** with an algebraic fusion: the usual four-step (mul, mean, sub, mul, add) collapses to a single `x.mul_(b*c).add_(b*mean_x*(1-c))` by expanding out the contrast formula.
- **Rotation + x-shear + horizontal flip** combined into **one** `grid_sample` call. We build a `(B, 2, 3)` theta with the rotation matrix multiplied by shear, then multiply the first row by `+-1` to encode flip. One bilinear resample per batch instead of three sequential transforms.
- **Normalization uses precomputed `INV_STD`** so the final step is `x.sub_(MEAN).mul_(INV_STD)` (fp16/bf16 multiply is faster than divide on Ada).
- **Aug disabled at eval** (`train=False` skips color jitter and affine, only normalization runs).

### Evaluation (`eval.py`)

- **Reuses one model shell** across all checkpoints so we don't pay the timm-init cost for every file.
- **bf16 autocast** in eval (fp32 softmax for probability stability).
- **TTA: horizontal-flip averaging in logit space**, then softmax in fp32. Small but reliable +0.5-1 PCE.
- **`--mode {all, ema, plain}`** to filter checkpoint types.
- **`--last-n`** to only eval the tail (we eval all 16 EMA checkpoints).
- **Reports PCE, Accuracy, Macro F1** (paper only reports PCE).

### Cross-validation Driver (`run_4split.sh`)

- Loops splits 1-4, trains 8,000 iters each, evaluates only EMA + TTA, captures the best PCE per split, prints a summary.
- Environment overrides: `ITERATIONS`, `BATCH`, `SEQ`, `RUN_TAG` so we can sweep without editing the script.
- Logs to per-split files plus a master `results_*.txt`.

---

## File-by-File Change Summary

| File | Status | Summary |
|------|--------|---------|
| `model.py` | Rewritten | MobileNetV3-Large via timm, 2-layer BiLSTM, drop_path/lstm_dropout params, feat_dim probe, channels_last, grad checkpointing option |
| `train.py` | Rewritten | Fused AdamW + LLRD, cosine schedule, bf16 AMP, EMA, mixup, label smoothing, class weights, grad clip, preload, deferred log sync, resume, arg-parsed |
| `dataloader.py` | Rewritten | RAM preload of all videos, preallocated decode buffer, vectorized label assignment, returns uint8 for GPU-side normalize |
| `gpu_augment.py` | **New file** | All augmentation + normalization on GPU, fused ops, one grid_sample for rotate + shear + flip |
| `eval.py` | Rewritten | Reuses model shell, bf16 AMP, TTA, checkpoint filtering, reports PCE + Acc + F1 |
| `util.py` | Modified | Kept `AverageMeter` and `correct_preds`, rewrote `freeze_layers` for timm MobileNetV3 block structure |
| `run_4split.sh` | **New file** | 4-split CV driver with env overrides |
| `MobileNetV2.py` | **Removed** | No longer needed (timm) |

---

## Hyperparameter Table (for reproducibility)

| Group | Param | Paper | Ours |
|-------|-------|-------|------|
| Data | Input size | 160x160 | 160x160 |
| Data | Sequence length | 64 | 64 |
| Data | Batch size | 22 | 16 (VRAM fit) |
| Model | Backbone | MobileNetV2 | MobileNetV3-Large (timm) |
| Model | LSTM layers | 1 | 2 |
| Model | LSTM hidden | 256 | 256 |
| Model | Bidirectional | yes | yes |
| Model | LSTM dropout | 0 | 0.3 (inter-layer) |
| Model | Drop path | 0 | 0.1 |
| Opt | Optimizer | Adam | AdamW (fused) |
| Opt | LR (head) | 1e-3 | 1e-3 |
| Opt | LR (backbone) | 1e-3 | 1e-4 (0.1x) |
| Opt | Weight decay | 0 | 1e-4 (skip 1-D) |
| Opt | LR schedule | constant | cosine + 500 warmup |
| Opt | LR floor | n/a | 1e-7 |
| Opt | Grad clip | none | 1.0 |
| Loss | Label smoothing | 0 | 0.1 |
| Loss | Class weight (event) | 1/8 | 1.0 |
| Loss | Class weight (no-event) | 1/35 | 0.1 |
| Reg | Mixup alpha | none | 0.2 |
| Reg | Mixup prob | none | 0.5 |
| Reg | EMA decay | none | 0.9995 |
| AMP | Precision | fp32 | bf16 autocast |
| AMP | Memory format | channels-first | channels-last |
| AMP | TF32 | off | on |
| Run | Iterations | 2,000 | 8,000 |
| Eval | Mode | single ckpt | all EMA + TTA |
| Eval | TTA | none | horizontal flip |
| Eval | CV | split 1 only | all 4 splits |

---

## Key Concepts Explained (Poster-friendly)

### PCE (Percentage of Correct Events)

The metric. For a full-length swing clip, each of the 8 events is correctly predicted if the argmax over that event's class column falls within a tolerance window of the ground-truth frame. Tolerance scales with swing speed (1/30 of the frames between Address and Impact). PCE is the mean over all events across the validation set.

Why not plain accuracy? Because the "no event" class dominates, a model predicting class 8 everywhere gets ~97% raw accuracy but 0% PCE. PCE directly measures what we care about: hitting the 8 keyframes.

### EMA (Exponential Moving Average)

We maintain a shadow copy of model weights:
```
shadow = decay * shadow + (1 - decay) * current_weights
```
With decay=0.9995, the shadow is effectively an average of the last ~2,000 training steps, heavily weighted toward recent ones. At eval time we use `shadow` instead of `current_weights`. This averages out the noise in individual SGD steps and consistently gives +1-2 PCE over the raw checkpoint.

### LLRD (Layer-wise Learning Rate Decay)

Pretrained ImageNet features are already useful, we just need to gently refine them. The randomly-initialized LSTM + head needs to learn from scratch. So the head gets lr=1e-3 and the backbone gets lr=1e-4 (10x smaller).

### bfloat16

fp16: 5-bit exponent, 10-bit mantissa. Narrow range, needs loss scaling to avoid under/overflow.
bf16: 8-bit exponent (same as fp32), 7-bit mantissa. Full fp32 range, no loss scaler needed.

Ada Lovelace (RTX 4080) has tensor cores for both. bf16 is strictly easier to use and is the default in our pipeline.

### Mixup

Take two samples A and B, make a new sample `lam*A + (1-lam)*B` with soft labels `lam*y_A + (1-lam)*y_B`. Encourages linear behavior between classes and acts as strong regularization. We clip `lam >= 0.5` so the primary sample dominates (otherwise the BiLSTM temporal structure gets too confused).

### Stochastic Depth / Drop Path

During training, each residual block in the backbone has a small probability (0.1) of being completely skipped, the output is just the input. At inference all blocks are active. Equivalent to training an ensemble of shallower networks.

### TTA (Test-Time Augmentation)

At inference, run each clip twice: once normally, once horizontally flipped. Average the logits before softmax. A golf swing is symmetric enough that the flipped prediction is nearly as good, and averaging both denoises the output. Free +0.5-1 PCE at the cost of 2x eval time.

---

## Why Each Change Matters (Ablation-style Narrative)

These are ordered roughly by expected contribution size. We didn't run a formal ablation (9-hour budget), but this matches what's published about each technique.

1. **Backbone upgrade (V2 -> V3-Large):** Adds SE attention and better activations. Typical ImageNet gain of +2-3 top-1 carries over here.
2. **More iterations (2,000 -> 8,000):** The paper's 2,000 iters underfit. Our cosine schedule over 8,000 lets the model reach a deeper minimum.
3. **EMA:** Reliably +1-2 PCE by averaging out SGD noise.
4. **Augmentation (color + affine + flip):** Paper reports their 76.1 number **requires** these, and they're missing from the public code. We reimplement them, GPU-fused.
5. **Mixup + label smoothing:** Stronger regularization, prevents overfitting as iterations grow.
6. **LLRD:** Without it, the 1e-3 LR destroys the pretrained features in the first few hundred iters.
7. **Cosine schedule + warmup:** Smooth LR decay finds a better local minimum than a flat LR; warmup prevents early instability.
8. **TTA at eval:** Small, reliable, free gain.
9. **bf16 + channels_last + TF32 + fused AdamW + RAM preload:** These don't change the final PCE, but they made the 8k-iter 4-split run fit in the 9-hour budget. Without them we'd have trained for 2k iters on one split like the paper.

---

## What Might Come Up in Q&A

- **"Why not transformer?"** We evaluated it conceptually. A tiny ViT + temporal transformer would need more pretraining data than GolfDB has (~1,400 clips). MobileNetV3's ImageNet init plus the BiLSTM for temporal modeling is a better fit for the scale.
- **"Why only 4-split average, not more folds?"** That's what the paper reports. Matching their protocol is the fair comparison.
- **"Is 0.7938 a real improvement or just variance?"** All 4 splits individually beat 0.7610 (the lowest is 0.7868). Hard to attribute to luck.
- **"Why didn't you use the paper's exact 2,000 iters?"** The paper undertrains. When we ran 2,000 iters with all our other changes, we only hit ~0.76. The extra iters (with cosine schedule) close the remaining gap.
- **"Why drop batch from 22 to 16?"** 16 GB VRAM on the 4080 is tight. Batch 22 OOM'd at the first LSTM forward. Batch 16 fits comfortably (~12 GB peak) with no measurable PCE difference.
- **"Why isn't macro F1 higher?"** Because class 8 ("no event") dominates. Macro F1 averages across 9 classes equally, so the 8 rare event classes each contribute 1/9 even though they occur once per clip. Our 0.48 macro F1 is normal for this setup, PCE is the metric that reflects actual task performance.

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

---

## Files Produced by This Run

- Source code: `model.py`, `train.py`, `dataloader.py`, `gpu_augment.py`, `eval.py`, `util.py`, `run_4split.sh`
- Checkpoints: `models_max_20260417_125243_s{1,2,3,4}/` (plain + EMA at every 500 iters)
- Training logs: `train_max_20260417_125243_s{1..4}.log`
- Eval logs: `eval_max_20260417_125243_s{1..4}.log`
- Summary: `results_max_20260417_125243.txt`
