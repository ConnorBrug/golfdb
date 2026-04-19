import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import GolfDB
from gpu_augment import augment_and_normalize
from model import EventDetector
from util import correct_preds


def build_model(lstm_layers, lstm_hidden, lstm_dropout, drop_path):
    model = EventDetector(
        pretrain=False,
        width_mult=1.0,
        lstm_layers=lstm_layers,
        lstm_hidden=lstm_hidden,
        bidirectional=True,
        dropout=False,
        cnn_dropout=0.0,
        drop_path_rate=drop_path,
        lstm_dropout=lstm_dropout,
        checkpoint_backbone=False,
    )
    model = model.cuda()
    model = model.to(memory_format=torch.channels_last)
    model.eval()
    return model


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, split, seq_length, num_workers, vid_dir,
                        lstm_layers, lstm_hidden, lstm_dropout, drop_path,
                        tta, amp_dtype, model=None, disp=False):
    dataset = GolfDB(
        data_file=f'data/val_split_{split}.pkl',
        vid_dir=vid_dir,
        seq_length=seq_length,
        train=False,
    )

    loader_kwargs = dict(
        dataset=dataset, batch_size=1, shuffle=False, num_workers=num_workers,
        drop_last=False, pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2
    data_loader = DataLoader(**loader_kwargs)

    if model is None:
        model = build_model(lstm_layers, lstm_hidden, lstm_dropout, drop_path)

    save_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()

    correct = []
    all_preds, all_labels = [], []

    for idx, sample in enumerate(tqdm(data_loader, desc=os.path.basename(ckpt_path), leave=False)):
        images, labels = sample['images'], sample['labels']
        total_frames = images.shape[1]

        probs = []
        batch = 0
        while batch * seq_length < total_frames:
            start = batch * seq_length
            end = min((batch + 1) * seq_length, total_frames)
            image_batch_u8 = images[:, start:end, :, :, :].cuda(non_blocking=True)

            with autocast('cuda', dtype=amp_dtype):
                image_batch = augment_and_normalize(image_batch_u8, train=False, dtype=amp_dtype)
                logits = model(image_batch)
                if tta:
                    flipped = torch.flip(image_batch, dims=[-1])
                    logits_f = model(flipped)
                    logits = 0.5 * (logits + logits_f)

            p = F.softmax(logits.float(), dim=1)
            probs.append(p.cpu().numpy())
            batch += 1

        probs = np.concatenate(probs, axis=0)
        _, _, _, _, c = correct_preds(probs, labels.squeeze().numpy())
        if disp:
            print(idx, c)
        correct.append(c)

        frame_preds = np.argmax(probs, axis=1)
        frame_labels = labels.squeeze().numpy()
        all_preds.extend(frame_preds.tolist())
        all_labels.extend(frame_labels.tolist())

    pce = np.mean(correct)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return pce, accuracy, macro_f1


def select_checkpoints(model_dir, mode, last_n):
    all_ckpts = sorted(glob.glob(os.path.join(model_dir, 'swingnet_*.pth.tar')),
                       key=lambda p: int(os.path.basename(p).split('_')[-1].split('.')[0]))
    if not all_ckpts:
        return []

    is_ema = lambda p: 'swingnet_ema_' in os.path.basename(p)

    if mode == 'ema':
        picked = [p for p in all_ckpts if is_ema(p)]
    elif mode == 'plain':
        picked = [p for p in all_ckpts if not is_ema(p)]
    else:
        picked = all_ckpts

    if last_n > 0:
        picked = picked[-last_n:]
    return picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', nargs='?', default='models')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--vid-dir', type=str, default='data/videos_160/')
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--lstm-hidden', type=int, default=256)
    parser.add_argument('--lstm-dropout', type=float, default=0.3)
    parser.add_argument('--drop-path', type=float, default=0.1)
    parser.add_argument('--tta', action='store_true', help='Horizontal flip averaging')
    parser.add_argument('--mode', choices=['all', 'ema', 'plain'], default='all',
                        help='Which checkpoints to evaluate')
    parser.add_argument('--last-n', type=int, default=0,
                        help='After filtering, keep only the last N checkpoints (0 = keep all)')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 autocast instead of default bf16')
    args = parser.parse_args()

    checkpoints = select_checkpoints(args.model_dir, args.mode, args.last_n)
    if not checkpoints:
        print(f'No checkpoints found in {args.model_dir}/')
        return

    print(f'Evaluating {len(checkpoints)} checkpoint(s) on split {args.split}. '
          f'TTA={args.tta}, mode={args.mode}, last_n={args.last_n}\n')

    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    model = build_model(args.lstm_layers, args.lstm_hidden, args.lstm_dropout, args.drop_path)

    results = []

    for ckpt_path in checkpoints:
        pce, acc, f1 = evaluate_checkpoint(
            ckpt_path, split=args.split, seq_length=args.seq_length,
            num_workers=args.num_workers, vid_dir=args.vid_dir,
            lstm_layers=args.lstm_layers, lstm_hidden=args.lstm_hidden,
            lstm_dropout=args.lstm_dropout, drop_path=args.drop_path,
            tta=args.tta, amp_dtype=amp_dtype, model=model, disp=False,
        )
        ckpt_name = os.path.basename(ckpt_path)
        print(f'{ckpt_name}: PCE = {pce:.4f}  Accuracy = {acc:.4f}  Macro F1 = {f1:.4f}')
        results.append((ckpt_name, pce, acc, f1))

    results_sorted = sorted(results, key=lambda r: r[1], reverse=True)
    ema_results = [r for r in results if 'swingnet_ema_' in r[0]]
    raw_results = [r for r in results if 'swingnet_ema_' not in r[0]]

    print('\n' + '=' * 70)
    print('Top 5 checkpoints by PCE:')
    for name, pce, acc, f1 in results_sorted[:5]:
        print(f'  {name:35s}  PCE={pce:.4f}  Acc={acc:.4f}  F1={f1:.4f}')

    if ema_results and raw_results:
        best_ema = max(ema_results, key=lambda r: r[1])
        best_raw = max(raw_results, key=lambda r: r[1])
        print('\nBest EMA vs best raw:')
        print(f'  EMA: {best_ema[0]:35s}  PCE={best_ema[1]:.4f}')
        print(f'  RAW: {best_raw[0]:35s}  PCE={best_raw[1]:.4f}')
        print(f'  Gain from EMA: {best_ema[1] - best_raw[1]:+.4f}')

    best = results_sorted[0]
    print(f'\nBest checkpoint: {best[0]}')
    print(f'  PCE:       {best[1]:.4f}')
    print(f'  Accuracy:  {best[2]:.4f}')
    print(f'  Macro F1:  {best[3]:.4f}')
    print('\nPaper baseline (MobileNetV2): PCE = 0.7610')


if __name__ == '__main__':
    main()
