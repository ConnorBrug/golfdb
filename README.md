# GolfDB SwingNet

Reimplementation and optimization of the SwingNet baseline from McNally et al., *"GolfDB: A Video Database for Golf Swing Sequencing"* (CVPR Workshops 2019). Produced for CS 4267 (Deep Learning), Spring 2026.

**4-split cross-validation PCE: 0.7938**
(paper baseline: 0.7610, a +3.3 pp / +4.3% improvement)

See [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) for a full comparison to the paper, hyperparameter table, concept explainers, and poster-ready notes.

---

## What's Different vs the Paper

- MobileNetV3-Large backbone (was MobileNetV2), via `timm`
- 2-layer BiLSTM with inter-layer dropout (was 1-layer, no dropout)
- bf16 mixed precision + channels-last + TF32 on Ada tensor cores
- Fused AdamW with Layer-wise Learning Rate Decay (backbone at 0.1x head LR)
- Cosine LR schedule + 500-iter warmup (was constant LR)
- Mixup (alpha=0.2, p=0.5), label smoothing (0.1), stochastic depth (0.1)
- EMA of model weights (decay=0.9995) for a smoother final model
- Horizontal-flip TTA at evaluation
- Full-dataset RAM preload, all augmentation on GPU
- 4-split cross-validation (paper only reports split 1)
- 8,000 training iterations per split with cosine decay (was 2,000)

Full breakdown in `PROJECT_SUMMARY.md`.

---

## Repository Layout

```
golfdb/
  model.py              # MobileNetV3 + BiLSTM + linear head
  train.py              # Training loop, AMP, EMA, mixup, LLRD, cosine LR
  eval.py               # Checkpoint sweep, TTA, PCE / accuracy / F1
  dataloader.py         # RAM-preload dataset, GPU-ready uint8 tensors
  gpu_augment.py        # Fused color jitter + rotate/shear/flip + normalize
  util.py               # AverageMeter, PCE metric, layer freezing
  run_4split.sh         # 4-split cross-validation driver
  test_video.py         # Run the model on a single mp4 for demo
  test_video.mp4        # Example swing video

  data/
    generate_splits.py  # Build the 4 CV splits from the .mat file
    preprocess_videos.py
    golfDB.mat          # Raw GolfDB annotations (~1,400 swings)
    golfDB.pkl          # Pickled dataframe used by the splits
    videos_160/         # 160x160 pre-cropped swing clips (not in git, 13 GB)

  results/              # Final 4-split PCE summaries
  PROJECT_SUMMARY.md    # Detailed writeup, poster reference
  README.md             # This file
  .gitignore
```

Not in git (generated at runtime): `.venv/`, `__pycache__/`, `models_*/`, `logs/`, `data/videos_160/`.

---

## Requirements

- Python 3.10+
- CUDA-capable GPU (tested on RTX 4080 16 GB)
- `torch>=2.0`, `timm>=1.0`, `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `tqdm`, `einops`, `scipy`

Install:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
# .venv\Scripts\activate         # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm opencv-python numpy pandas scikit-learn tqdm einops scipy
```

---

## Data Setup

1. Download the pre-cropped 160x160 videos from the [original repo](https://github.com/wmcnally/golfdb) (Google Drive link in their README) and extract to `data/videos_160/`.
2. The `.pkl` split files (`data/train_split_{1..4}.pkl`, `data/val_split_{1..4}.pkl`) are generated from `data/golfDB.mat`. If they aren't already present:
   ```bash
   cd data && python generate_splits.py && cd ..
   ```

---

## Training

Single split:

```bash
python train.py --split 1 --model-dir models_s1 --batch-size 16 --seq-length 64 --iterations 8000
```

All 4 splits (our final experiment setup):

```bash
bash run_4split.sh
```

Environment overrides the driver script understands: `BATCH`, `SEQ`, `ITERATIONS`, `RUN_TAG`.

Example:

```bash
BATCH=16 ITERATIONS=8000 bash run_4split.sh
```

The driver writes per-split training logs, eval logs, and a summary `results_<tag>.txt`.

---

## Evaluation

Evaluate EMA checkpoints with horizontal-flip TTA (matches our reported numbers):

```bash
python eval.py models_s1 --split 1 --seq-length 64 --mode ema --tta
```

Useful flags:

- `--mode {all, ema, plain}`: which checkpoints to evaluate
- `--last-n N`: after filtering, keep only the final N checkpoints
- `--tta`: horizontal-flip test-time augmentation
- `--fp16`: use fp16 autocast instead of default bf16

---

## Demo on a Single Video

```bash
python test_video.py -p test_video.mp4 --ckpt models_s1/swingnet_ema_8000.pth.tar
```

Prints the predicted frame index for each of the 8 swing events (Address, Toe-up, Mid-backswing, Top, Mid-downswing, Impact, Mid-follow-through, Finish) and shows the frame for each.

---

## Results

Run: `max_20260417_125243`, 4 splits x 8,000 iterations, batch=16, seq=64, bf16.

| Split | PCE | Accuracy | Macro F1 |
|-------|------|----------|----------|
| 1 | 0.7964 | 0.9500 | 0.4865 |
| 2 | 0.7957 | 0.9489 | 0.4705 |
| 3 | 0.7868 | 0.9480 | 0.4733 |
| 4 | 0.7964 | 0.9521 | 0.4788 |
| **Avg** | **0.7938** | **0.9498** | **0.4773** |
| Paper | 0.7610 | &mdash; | &mdash; |

Full results file: [`results/results_max_20260417_125243.txt`](./results/results_max_20260417_125243.txt).

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
