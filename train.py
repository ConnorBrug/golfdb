import argparse
import glob
import math
import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm

from dataloader import GolfDB, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter, RandomRotation
from model import EventDetector
from util import AverageMeter, freeze_layers


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    # Avoid thread oversubscription from OpenCV inside multiple workers.
    cv2.setNumThreads(0)
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(args):
    train_transforms = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
        RandomRotation(degrees=5),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    dataset = GolfDB(
        data_file=f'data/train_split_{args.split}.pkl',
        vid_dir=args.vid_dir,
        seq_length=args.seq_length,
        transform=train_transforms,
        train=True,
    )

    kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=seed_worker,
    )

    if args.num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = 2

    return DataLoader(**kwargs)


def build_model(args):
    model = EventDetector(
        pretrain=True,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
        cnn_dropout=0.0,
        checkpoint_backbone=args.grad_ckpt,
    )
    freeze_layers(args.k, model)
    model = model.cuda()
    model.train()
    return model


def save_ckpt(path, model, optimizer, scaler, iteration, args):
    torch.save({
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'args': vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--iterations', type=int, default=6000)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=36)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--k', type=int, default=5, help='number of backbone layers to freeze')
    parser.add_argument('--lr', type=float, default=8.2e-4)
    parser.add_argument('--lr-min', type=float, default=1e-6)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vid-dir', type=str, default='data/videos_160/')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--grad-ckpt', action='store_true',
                        help='Enable CNN activation checkpointing. Slower but lower VRAM.')
    args = parser.parse_args()

    seed_everything(args.seed)

    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    model = build_model(args)
    data_loader = build_loader(args)

    class_weights = torch.tensor([1/8] * 8 + [1/35], dtype=torch.float32, device='cuda')
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler('cuda')
    losses = AverageMeter()

    os.makedirs(args.model_dir, exist_ok=True)

    start_iter = 0
    if args.resume:
        existing = sorted(glob.glob(os.path.join(args.model_dir, 'swingnet_*.pth.tar')))
        if existing:
            latest = max(existing, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
            ckpt = torch.load(latest, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scaler_state_dict' in ckpt:
                scaler.load_state_dict(ckpt['scaler_state_dict'])
            start_iter = int(ckpt.get('iteration', 0))
            model.train()
            print(f'Resuming from {os.path.basename(latest)} (iteration {start_iter})')

    # Helpful presets for your box:
    #   Fast target (~1-2h if dataloader behaves): --k 5 --batch-size 36 --iterations 4600 --num-workers 8
    #   Max target (longer):                       --k 0 --batch-size 28 --iterations 7000 --num-workers 8 --grad-ckpt
    i = start_iter
    pbar = tqdm(total=args.iterations, initial=start_iter, desc='Training', unit='it')

    while i < args.iterations:
        for sample in data_loader:
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * i / args.iterations))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            images = sample['images'].cuda(non_blocking=True)
            labels = sample['labels'].cuda(non_blocking=True)

            with autocast('cuda', dtype=torch.float16):
                logits = model(images)
                labels = labels.reshape(args.batch_size * args.seq_length)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=f'{losses.val:.4f}', avg=f'{losses.avg:.4f}', lr=f'{lr:.6f}')
            pbar.update(1)
            i += 1

            if i % args.save_every == 0:
                save_ckpt(
                    os.path.join(args.model_dir, f'swingnet_{i}.pth.tar'),
                    model, optimizer, scaler, i, args
                )

            if i >= args.iterations:
                break

    pbar.close()


if __name__ == '__main__':
    main()
