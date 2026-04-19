"""Training loop for SwingNet.

Features:
  - bfloat16 autocast by default (fp16 via --fp16)
  - Fused AdamW with four-group Layer-wise LR Decay (backbone vs head, weight decay
    disabled on BN/bias/LSTM 1-D params)
  - Cosine LR schedule with linear warmup
  - Mixup (alpha=0.2, p=0.5), label smoothing (0.1), stochastic depth (0.1)
  - EMA of model weights saved every --save-every iterations
  - Deferred .item() loss sync to minimize CUDA stalls
  - Resume from the latest non-EMA checkpoint with --resume

Launch:
  python train.py --split 1 --model-dir models_s1 --batch-size 16 \
      --seq-length 64 --iterations 8000
"""

import argparse
import glob
import math
import os
import random

# set before importing torch so the allocator honors it
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import GolfDB
from gpu_augment import augment_and_normalize
from model import EventDetector
from util import AverageMeter, freeze_layers


class EMA:
    """Exponential moving average of model weights, held in fp32 on-device for stability."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.detach().clone().float()

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                # shadow = d*shadow + (1-d)*p, fp32
                self.shadow[name].mul_(d).add_(p.data.float(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model):
        backup = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name].to(p.dtype))
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        for name, p in model.named_parameters():
            if name in backup:
                p.data.copy_(backup[name])

    def state_dict(self):
        return {k: v.cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd, device):
        self.shadow = {k: v.to(device).float() for k, v in sd.items()}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # opencv threads fight with dataloader workers, pin it to 0 inside each worker
    cv2.setNumThreads(0)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(args):
    dataset = GolfDB(
        data_file=f'data/train_split_{args.split}.pkl',
        vid_dir=args.vid_dir,
        seq_length=args.seq_length,
        train=True,
        preload=not args.no_preload,
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        pin_memory_device='cuda',
        worker_init_fn=seed_worker,
    )

    if args.num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 4

    return DataLoader(**kwargs)


def build_model(args):
    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=args.lstm_layers,
        lstm_hidden=args.lstm_hidden,
        bidirectional=True,
        dropout=False,
        cnn_dropout=0.0,
        drop_path_rate=args.drop_path,
        lstm_dropout=args.lstm_dropout,
        checkpoint_backbone=args.grad_ckpt,
    )
    freeze_layers(args.k, model)
    model = model.cuda()
    model = model.to(memory_format=torch.channels_last)
    model.train()
    if args.compile:
        model = torch.compile(model, mode='default', dynamic=False)
    return model


def build_param_groups(model, base_lr, backbone_lr_mult, weight_decay):
    """Four param groups: backbone/head x decay/no-decay.
    No-decay for 1-D params (BN, biases) is standard best practice and costs nothing."""
    target = model._orig_mod if hasattr(model, '_orig_mod') else model

    bb_decay, bb_nodecay, hd_decay, hd_nodecay = [], [], [], []
    for name, p in target.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = name.startswith('cnn.')
        # 1-D tensors are BN weights, biases, LSTM biases — all should skip weight decay
        is_nodecay = p.dim() <= 1 or name.endswith('.bias') or 'bias_ih' in name or 'bias_hh' in name
        if is_backbone and is_nodecay:
            bb_nodecay.append(p)
        elif is_backbone:
            bb_decay.append(p)
        elif is_nodecay:
            hd_nodecay.append(p)
        else:
            hd_decay.append(p)

    groups = [
        {'params': bb_decay, 'lr': base_lr * backbone_lr_mult, 'weight_decay': weight_decay},
        {'params': bb_nodecay, 'lr': base_lr * backbone_lr_mult, 'weight_decay': 0.0},
        {'params': hd_decay, 'lr': base_lr, 'weight_decay': weight_decay},
        {'params': hd_nodecay, 'lr': base_lr, 'weight_decay': 0.0},
    ]
    print(f'Param groups: backbone decay={len(bb_decay)} no-decay={len(bb_nodecay)}, '
          f'head decay={len(hd_decay)} no-decay={len(hd_nodecay)}')
    return groups


def mixup_sequences(images, labels, alpha, num_classes=9):
    """Mix two sequences at the batch level. Labels become soft one-hot mixes.
    images: (B, T, C, H, W) float. labels: (B*T,) long. Returns images, soft_labels (B*T, C)."""
    B, T = images.shape[0], images.shape[1]
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep lam >= 0.5 so the primary signal dominates

    perm = torch.randperm(B, device=images.device)
    # one big fused op: lam*A + (1-lam)*B. torch.lerp on A with B and weight (1-lam).
    images_mixed = torch.lerp(images, images[perm], 1.0 - lam)

    labels_bt = labels.view(B, T)
    labels_a = torch.nn.functional.one_hot(labels_bt, num_classes).float()
    labels_b = torch.nn.functional.one_hot(labels_bt[perm], num_classes).float()
    labels_soft = torch.lerp(labels_a, labels_b, 1.0 - lam)
    labels_soft = labels_soft.view(B * T, num_classes)

    return images_mixed, labels_soft


def soft_cross_entropy(logits, soft_targets, class_weights, label_smoothing):
    """Cross-entropy against (B*T, C) soft targets with class weights and label smoothing.
    Matches nn.CrossEntropyLoss(weight=...) mean reduction semantics."""
    num_classes = logits.shape[1]
    if label_smoothing > 0:
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / num_classes

    log_probs = torch.nn.functional.log_softmax(logits, dim=1)
    weighted = log_probs * class_weights.view(1, -1)
    loss = -(soft_targets * weighted).sum(dim=1)
    target_weights = (soft_targets * class_weights.view(1, -1)).sum(dim=1)
    return loss.sum() / target_weights.sum().clamp_min(1e-8)


def save_ckpt(path, model, optimizer, scaler, iteration, args, ema=None):
    state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
    payload = {
        'iteration': iteration,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
        'args': vars(args),
    }
    if ema is not None:
        payload['ema_state_dict'] = ema.state_dict()
    torch.save(payload, path)


def save_ema_as_model(path, model, ema, iteration, args):
    """Save the EMA weights as a plain model checkpoint so eval.py can load it directly."""
    target = model._orig_mod if hasattr(model, '_orig_mod') else model
    backup = ema.apply_to(target)
    state_dict = target.state_dict()
    torch.save({'iteration': iteration, 'model_state_dict': state_dict, 'args': vars(args)}, path)
    ema.restore(target, backup)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=8000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=22)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--k', type=int, default=0, help='Number of backbone layers to freeze')
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--lstm-hidden', type=int, default=256)
    parser.add_argument('--lstm-dropout', type=float, default=0.3)
    parser.add_argument('--drop-path', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr-min', type=float, default=1e-7)
    parser.add_argument('--backbone-lr-mult', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--label-smoothing', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--mixup-prob', type=float, default=0.5)
    parser.add_argument('--ema-decay', type=float, default=0.9995)
    parser.add_argument('--warmup-iters', type=int, default=500)
    parser.add_argument('--log-every', type=int, default=10, help='Iterations between loss syncs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vid-dir', type=str, default='data/videos_160/')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--grad-ckpt', action='store_true')
    parser.add_argument('--no-preload', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 instead of default bf16 (needs GradScaler)')
    args = parser.parse_args()

    seed_everything(args.seed)

    # TF32 + cudnn benchmark for max matmul/conv throughput on Ada
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    model = build_model(args)
    data_loader = build_loader(args)

    # class 8 is "no event", downweighted heavily to combat the ~30:1 imbalance
    class_weights = torch.tensor([1.0] * 8 + [0.1], dtype=torch.float32, device='cuda')

    param_groups = build_param_groups(model, args.lr, args.backbone_lr_mult, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, fused=True)
    base_lrs = [g['lr'] for g in optimizer.param_groups]

    # default bf16 on Ada: no loss scaler needed, full range, faster than fp16 in practice
    use_bf16 = not args.fp16
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = not use_bf16
    scaler = GradScaler('cuda', enabled=use_scaler) if use_scaler else None

    ema = EMA(model._orig_mod if hasattr(model, '_orig_mod') else model, args.ema_decay) if args.ema_decay > 0 else None

    os.makedirs(args.model_dir, exist_ok=True)

    start_iter = 0
    if args.resume:
        existing = sorted(glob.glob(os.path.join(args.model_dir, 'swingnet_*.pth.tar')))
        existing = [p for p in existing if 'swingnet_ema_' not in os.path.basename(p)]
        if existing:
            latest = max(existing, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
            ckpt = torch.load(latest, map_location='cpu', weights_only=False)
            target = model._orig_mod if hasattr(model, '_orig_mod') else model
            target.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if scaler is not None and ckpt.get('scaler_state_dict') is not None:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            if ema is not None and 'ema_state_dict' in ckpt:
                ema.load_state_dict(ckpt['ema_state_dict'], device='cuda')
            start_iter = int(ckpt.get('iteration', 0))
            model.train()
            print(f'Resuming from {os.path.basename(latest)} (iteration {start_iter})')

    losses = AverageMeter()
    # deferred-sync loss accumulator: keep loss on GPU, .item() only every log_every iters
    loss_acc = torch.zeros((), device='cuda', dtype=torch.float32)
    n_acc = 0

    i = start_iter
    pbar = tqdm(total=args.iterations, initial=start_iter, desc='Training', unit='it', mininterval=0.5)

    while i < args.iterations:
        for sample in data_loader:
            # cosine schedule with linear warmup, computed once per iter in pure Python
            if i < args.warmup_iters:
                scale = (i + 1) / args.warmup_iters
            else:
                progress = (i - args.warmup_iters) / max(1, args.iterations - args.warmup_iters)
                cos = 0.5 * (1 + math.cos(math.pi * progress))
                floor_frac = args.lr_min / args.lr
                scale = floor_frac + (1 - floor_frac) * cos
            for pg, base in zip(optimizer.param_groups, base_lrs):
                pg['lr'] = base * scale

            images_u8 = sample['images'].cuda(non_blocking=True)
            labels = sample['labels'].cuda(non_blocking=True).reshape(-1)

            with autocast('cuda', dtype=amp_dtype):
                images = augment_and_normalize(images_u8, train=True, dtype=amp_dtype)
                del images_u8

                do_mixup = args.mixup_alpha > 0 and random.random() < args.mixup_prob
                if do_mixup:
                    images, soft_labels = mixup_sequences(images, labels, args.mixup_alpha)
                    logits = model(images)
                    loss = soft_cross_entropy(logits, soft_labels, class_weights, args.label_smoothing)
                else:
                    logits = model(images)
                    loss = torch.nn.functional.cross_entropy(
                        logits, labels, weight=class_weights, label_smoothing=args.label_smoothing
                    )

            optimizer.zero_grad(set_to_none=True)
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()

            if ema is not None:
                ema.update(model._orig_mod if hasattr(model, '_orig_mod') else model)

            # deferred sync: keep the tensor on GPU, only .item() every log_every iters
            n = images.size(0)
            loss_acc += loss.detach().float() * n
            n_acc += n
            i += 1
            pbar.update(1)

            if (i % args.log_every) == 0:
                avg = (loss_acc / max(n_acc, 1)).item()
                losses.update(avg, n_acc)
                loss_acc.zero_()
                n_acc = 0
                head_lr = optimizer.param_groups[-1]['lr']
                pbar.set_postfix(loss=f'{losses.val:.4f}', avg=f'{losses.avg:.4f}',
                                 lr=f'{head_lr:.6f}', refresh=False)

            if i % args.save_every == 0:
                save_ckpt(os.path.join(args.model_dir, f'swingnet_{i}.pth.tar'),
                          model, optimizer, scaler, i, args, ema=ema)
                if ema is not None:
                    save_ema_as_model(os.path.join(args.model_dir, f'swingnet_ema_{i}.pth.tar'),
                                      model, ema, i, args)

            if i >= args.iterations:
                break

    pbar.close()

    # always save a final checkpoint, even if iterations % save_every != 0
    if i % args.save_every != 0:
        save_ckpt(os.path.join(args.model_dir, f'swingnet_{i}.pth.tar'),
                  model, optimizer, scaler, i, args, ema=ema)
        if ema is not None:
            save_ema_as_model(os.path.join(args.model_dir, f'swingnet_ema_{i}.pth.tar'),
                              model, ema, i, args)


if __name__ == '__main__':
    main()
