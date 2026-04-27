# Poster Breakdown — Task Split & Differentiation Plan

**Team:** Bobby Miller, Connor Brugger  
**Course:** CS 4267 Deep Learning, Spring 2026  
**Final PCE:** 0.7938 (4-split avg) vs. paper 0.7610 (+3.3 pp)

---

## Who Did What

### Bobby — Phase 1: Architecture & Core Implementation

Got the pipeline off the ground. Responsible for anything needed to make the model *run at all*:

- **Model architecture** ([model.py](./model.py)): Implemented the CNN + BiLSTM + linear head from the GolfDB paper. Upgraded the backbone from MobileNetV2 → MobileNetV3-Large via `timm`. Added the second BiLSTM layer (paper used 1) with inter-layer dropout (0.3). Added the `head_hidden_size` probe so the model doesn't crash on timm 1.0+ (which changed `num_features` semantics).
- **Making it train stably**: The things that were necessary plumbing to get training to converge at all, not just "go faster":
  - Layer-wise Learning Rate Decay (LLRD) — backbone at 0.1× head LR. Without this, the head's 1e-3 LR destroyed pretrained ImageNet features in the first few hundred iterations.
  - Cosine LR schedule with 500-iteration linear warmup. A flat LR plateaued around 0.76; cosine + warmup was needed to reach 0.79+.
  - Class weight adjustment: moved from paper's `1/35` on the no-event class to `0.1`. The original weight over-suppressed class 8 once label smoothing and mixup were in the mix.
  - Label smoothing 0.1 and Mixup (α=0.2, p=0.5) — regularization decisions paired with the class-weight change.
  - Stochastic depth (`drop_path=0.1`) on the backbone.

### Connor — Phase 2: Optimization & Throughput

Once the model was running, took it fast and repeatable:

- **GPU augmentation** ([gpu_augment.py](./gpu_augment.py)): Wrote from scratch. Fused color jitter (brightness + contrast + saturation into one mul+add), rotation + shear + horizontal flip folded into a single `grid_sample`. Normalize uses precomputed `INV_STD` (multiply instead of divide). All on-GPU.
- **Mixed precision pipeline**: bf16 autocast by default on Ada Lovelace, LSTM kept in fp32 for stability, `channels_last` memory format for the CNN (~25–30% faster), TF32 matmul + `cudnn.benchmark`.
- **Fused AdamW**: four param groups (backbone/head × decay/no-decay), single-kernel updates.
- **Data pipeline**: Full-dataset RAM preload (~6–8 GB of uint8 frames); preallocated decode buffer; uint8 transport to GPU; persistent workers with prefetch factor 4.
- **EMA**: exponential moving average of weights (decay 0.9995); all eval uses the EMA copy.
- **Evaluation infrastructure** ([eval.py](./eval.py)): checkpoint sweep, horizontal-flip TTA, PCE + Accuracy + Macro F1 reporting. 4-split cross-validation driver (`run_4split.sh`). Results pipeline at `results/`.

---

## The Poster Problem

We need **fundamentally different posters** but have limited saved figures/data. Connor has the model checkpoint (`poster_best/swingnet_ema_8000.pth.tar`), making additional inference possible.

### What we have right now (no new runs needed)

- 4-split PCE / Accuracy / Macro F1 results: `results/results_max_20260417_125243.txt`
- Model checkpoint: `poster_best/swingnet_ema_8000.pth.tar`
- All source code and hyperparameter history in git

### What Bobby can generate with the checkpoint (Connor runs inference, Bobby analyzes)

Connor runs `eval.py` with a small patch to dump per-frame predictions and labels to a `.pkl` or `.csv`. Bobby then uses those outputs to produce:

1. **Per-class confusion matrix** — 9×9 heatmap, normalized by true label
2. **Per-class precision / recall / F1 table** — highlights which swing phases the model struggles with
3. **ROC curves** — one curve per event class (classes 0–7); AUC per class
4. **Class-imbalance visualization** — bar chart of label distribution (class 8 ≈ 35× more frequent)

> `eval.py` already concatenates `frame_preds` and `frame_labels` across the validation set — just need to save them before the metric computation discards them. ~5 lines of code.

---

## Poster Angles (non-overlapping)

### Connor's Poster — "Modernizing a 2019 baseline: +3.3 PCE from training-pipeline engineering"

**Thesis:** The paper's architecture is fine. What's holding it back is a 2019 training recipe. Replace the recipe and the same architecture family jumps +3.3 pp PCE.

**Narrative:**
1. Baseline: swing-phase event detection, PCE metric, 76.1 paper result
2. Six levers changed: backbone, optimizer (AdamW + LLRD + cosine + warmup), regularization (mixup + label smoothing + EMA + stochastic depth), AMP (bf16 + channels_last + TF32), data pipeline (RAM preload + GPU augmentation), eval (TTA + full-EMA sweep + 4-split CV)
3. Key plot: **4-split bar chart** — 0.7964 / 0.7957 / 0.7868 / 0.7964 vs. 0.7610
4. Wall-clock story: 43 min/split end-to-end; the throughput work is what made 8k×4 splits fit in the time budget
5. Single-video demo still from `test_video.py`

**Page 3 (Connor's own contributions):** gpu_augment.py, bf16/channels_last/TF32 pipeline, fused AdamW, RAM preload, EMA, 4-split CV driver, eval infrastructure.

**Headline metric:** PCE. De-emphasize Macro F1.

---

### Bobby's Poster — "Per-frame swing-phase classification: imbalance, calibration, and what PCE doesn't tell you"

**Thesis:** When you reframe event detection as a 9-way per-frame classification problem, the class imbalance (~1:35 event-to-no-event) becomes the central challenge. Macro F1, per-class ROC, and confusion matrices reveal failure modes that PCE hides.

**Narrative:**
1. Problem reframed: 9 classes per frame, ~1:35 imbalance between event and no-event frames
2. Why Macro F1 and per-class ROC matter on imbalanced multiclass data — and what they show that PCE alone hides
3. Class-weight study: paper's `1/35` vs. our `0.1` vs. no weighting — effect on per-class recall
4. Regularization decisions (label smoothing + mixup): what they do to per-class recall and calibration, not just aggregate PCE
5. **Confusion matrix** (9×9): which phases get confused (Mid-backswing ↔ Top; Mid-downswing ↔ Impact are the usual culprits)
6. **Per-class precision / recall / F1 table** + **ROC curves** for the 8 event classes
7. Error analysis: where the model fails and why

**Page 3 (Bobby's own contributions):** Architecture implementation (MobileNetV3 + 2-layer BiLSTM), class-weight and regularization experiments, frame-level metric methodology, confusion matrix and per-class ROC analysis (new plots generated post-training).

**Headline metric:** Macro F1, AUC-ROC, confusion matrix. De-emphasize wall-clock and throughput.

---

## Differentiation Summary

| Dimension | Connor | Bobby |
|-----------|--------|-------|
| Framing | Event-sequence detection | Per-frame classification |
| Headline metric | PCE (+3.3 pp) | Macro F1 / AUC-ROC |
| Biggest plot | 4-split bar chart vs. paper | Confusion matrix + ROC curves |
| Story spine | Training-recipe modernization | Imbalance handling + calibration |
| Own contributions | Pipeline, throughput, CV protocol | Architecture, class-weight decisions, metric analysis |
| What to de-emphasize | Confusion matrix, per-class F1 | Wall-clock, bf16/TF32 |

Both angles are honest descriptions of the same training run. Team results belong to the team; each poster interprets them through a different lens.

---

## Action Items

| # | Owner | Task |
|---|-------|------|
| 1 | ~~Connor~~ Bobby | ~~Patch `eval.py` to save `frame_preds` and `frame_labels` to file~~ Done — `--save-preds` flag added, saves per-frame `probs` (N×9), `preds`, and `labels` to a compressed `.npz` |
| 2 | Connor | Run inference and share the `.npz` with Bobby: `python eval.py poster_best --split 1 --mode ema --last-n 1 --tta --save-preds` → produces `poster_best/preds_split1_swingnet_ema_8000.npz` |
| 3 | Bobby | Load with `d = np.load(...); probs, preds, labels = d['probs'], d['preds'], d['labels']` — generate confusion matrix, per-class F1 table, ROC curves (`probs` needed for ROC) |
| 4 | Bobby | Generate class-distribution bar chart from any split's label file (no model needed) |
| 5 | Connor | Take a single-video demo frame with `test_video.py --show` for the poster still |
| 6 | Both | Cross-check page 3 attribution — make sure no two sentences describe the same contribution |
