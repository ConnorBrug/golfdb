import argparse
import glob
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloader import GolfDB, ToTensor, Normalize
from model import EventDetector
from util import correct_preds


def evaluate_checkpoint(ckpt_path, split, seq_length, num_workers, vid_dir, disp=False):
    dataset = GolfDB(
        data_file=f'data/val_split_{split}.pkl',
        vid_dir=vid_dir,
        seq_length=seq_length,
        transform=transforms.Compose([
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        train=False,
    )

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    data_loader = DataLoader(**loader_kwargs)

    model = EventDetector(
        pretrain=False,  # never redownload weights in eval
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
        cnn_dropout=0.0,
        checkpoint_backbone=False,
    )

    save_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(save_dict['model_state_dict'])
    model.cuda()
    model.eval()

    correct = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(data_loader, desc=os.path.basename(ckpt_path), leave=False)):
            images, labels = sample['images'], sample['labels']

            batch = 0
            probs = []
            while batch * seq_length < images.shape[1]:
                start = batch * seq_length
                end = min((batch + 1) * seq_length, images.shape[1])
                image_batch = images[:, start:end, :, :, :].cuda(non_blocking=True)

                logits = model(image_batch)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', nargs='?', default='models')
    parser.add_argument('--split', type=int, default=1)
    parser.add_argument('--seq-length', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--vid-dir', type=str, default='data/videos_160/')
    args = parser.parse_args()

    checkpoints = sorted(glob.glob(os.path.join(args.model_dir, 'swingnet_*.pth.tar')))
    if not checkpoints:
        print(f'No checkpoints found in {args.model_dir}/')
        return

    print(f'Found {len(checkpoints)} checkpoints. Evaluating all...\n')

    best_pce = -1.0
    best_ckpt = ''
    best_acc = 0.0
    best_f1 = 0.0

    for ckpt_path in checkpoints:
        pce, acc, f1 = evaluate_checkpoint(
            ckpt_path,
            split=args.split,
            seq_length=args.seq_length,
            num_workers=args.num_workers,
            vid_dir=args.vid_dir,
            disp=False,
        )
        ckpt_name = os.path.basename(ckpt_path)
        print(f'{ckpt_name}: PCE = {pce:.4f}  Accuracy = {acc:.4f}  Macro F1 = {f1:.4f}')

        if pce > best_pce:
            best_pce = pce
            best_ckpt = ckpt_name
            best_acc = acc
            best_f1 = f1

    print(f'\nBest checkpoint: {best_ckpt}')
    print(f'  PCE:       {best_pce:.4f}')
    print(f'  Accuracy:  {best_acc:.4f}')
    print(f'  Macro F1:  {best_f1:.4f}')
    print('\nPaper baseline (MobileNetV2): PCE = 0.7610')


if __name__ == '__main__':
    main()
