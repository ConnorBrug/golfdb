from dataloader import GolfDB, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter, RandomRotation
from model import EventDetector
from util import *
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from tqdm import tqdm
import math
import os
import glob
import sys


if __name__ == '__main__':

    # training configuration
    split = 1
    iterations = 8000
    it_save = 100  # save model every 100 iterations
    n_cpu = 0
    seq_length = 64
    bs = 22  # batch size
    k = 0  # no frozen layers — full fine-tuning

    # learning rate schedule (lower initial LR since all layers are trainable)
    lr_init = 0.0005
    lr_min = 1e-6

    # performance
    cudnn.benchmark = True

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          bidirectional=True,
                          dropout=False)
    freeze_layers(k, model)
    model.train()
    model.cuda()

    # data augmentation (applied before ToTensor/Normalize, consistently across sequence)
    # the original paper did NOT use augmentation — this is our addition
    train_transforms = transforms.Compose([
        RandomHorizontalFlip(p=0.5),
        ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        RandomRotation(degrees=5),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = GolfDB(data_file='data/train_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=train_transforms,
                     train=True)

    data_loader = DataLoader(dataset,
                             batch_size=bs,
                             shuffle=True,
                             num_workers=n_cpu,
                             drop_last=True,
                             pin_memory=True)

    # the 8 golf swing events are classes 0 through 7, no-event is class 8
    weights = torch.FloatTensor([1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/8, 1/35]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # weight decay for regularization
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr_init, weight_decay=1e-4)

    # mixed precision
    scaler = GradScaler()

    losses = AverageMeter()

    if not os.path.exists('models'):
        os.mkdir('models')

    # ── Resume from latest checkpoint if available ──
    start_iter = 0
    existing = sorted(glob.glob('models/swingnet_*.pth.tar'))
    if existing and '--resume' in sys.argv:
        # find the highest iteration checkpoint
        latest = max(existing, key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]))
        start_iter = int(os.path.basename(latest).split('_')[1].split('.')[0])
        print(f'Resuming from {os.path.basename(latest)} (iteration {start_iter})')
        ckpt = torch.load(latest)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.train()

    i = start_iter
    pbar = tqdm(total=iterations, initial=start_iter, desc='Training', unit='it')
    while i < iterations:
        for sample in data_loader:
            # cosine annealing learning rate
            lr = lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(math.pi * i / iterations))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            images = sample['images'].cuda(non_blocking=True)
            labels = sample['labels'].cuda(non_blocking=True)

            with autocast():
                logits = model(images)
                labels = labels.view(bs * seq_length)
                loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=f'{losses.val:.4f}', avg=f'{losses.avg:.4f}', lr=f'{lr:.6f}')
            pbar.update(1)
            i += 1
            if i % it_save == 0:
                torch.save({'optimizer_state_dict': optimizer.state_dict(),
                            'model_state_dict': model.state_dict()}, 'models/swingnet_{}.pth.tar'.format(i))
            if i == iterations:
                break
    pbar.close()
