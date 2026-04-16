from model import EventDetector
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import GolfDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
from sklearn.metrics import accuracy_score, f1_score
import glob
import os


def eval(model, split, seq_length, n_cpu, disp):
    """Evaluate model on validation split.
    Returns PCE, frame-level accuracy, and macro F1."""
    dataset = GolfDB(data_file='data/val_split_{}.pkl'.format(split),
                     vid_dir='data/videos_160/',
                     seq_length=seq_length,
                     transform=transforms.Compose([ToTensor(),
                                                   Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                     train=False)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=False)

    correct = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i, sample in enumerate(data_loader):
            images, labels = sample['images'], sample['labels']
            # full samples do not fit into GPU memory so evaluate sample in 'seq_length' batches
            batch = 0
            while batch * seq_length < images.shape[1]:
                if (batch + 1) * seq_length > images.shape[1]:
                    image_batch = images[:, batch * seq_length:, :, :, :]
                else:
                    image_batch = images[:, batch * seq_length:(batch + 1) * seq_length, :, :, :]
                logits = model(image_batch.cuda())
                if batch == 0:
                    probs = F.softmax(logits.data, dim=1).cpu().numpy()
                else:
                    probs = np.append(probs, F.softmax(logits.data, dim=1).cpu().numpy(), 0)
                batch += 1

            # PCE (event-level metric from the paper)
            _, _, _, _, c = correct_preds(probs, labels.squeeze())
            if disp:
                print(i, c)
            correct.append(c)

            # frame-level predictions for accuracy and F1
            frame_preds = np.argmax(probs, axis=1)
            frame_labels = labels.squeeze().numpy()
            all_preds.extend(frame_preds.tolist())
            all_labels.extend(frame_labels.tolist())

    PCE = np.mean(correct)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return PCE, accuracy, macro_f1


if __name__ == '__main__':

    import sys
    split = 1
    seq_length = 64
    n_cpu = 0

    cudnn.benchmark = True

    # optionally pass a models folder as argument: python eval.py models_k5_augmented
    model_dir = sys.argv[1] if len(sys.argv) > 1 else 'models'

    # find all checkpoints and evaluate each one
    checkpoints = sorted(glob.glob(os.path.join(model_dir, 'swingnet_*.pth.tar')))
    if not checkpoints:
        print('No checkpoints found in models/')
        exit()

    print(f'Found {len(checkpoints)} checkpoints. Evaluating all...\n')

    best_pce = 0.0
    best_ckpt = ''
    best_acc = 0.0
    best_f1 = 0.0

    for ckpt_path in checkpoints:
        model = EventDetector(pretrain=True,
                              width_mult=1.,
                              lstm_layers=1,
                              lstm_hidden=256,
                              bidirectional=True,
                              dropout=False)

        save_dict = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(save_dict['model_state_dict'])
        model.cuda()
        model.eval()

        PCE, acc, f1 = eval(model, split, seq_length, n_cpu, False)
        ckpt_name = os.path.basename(ckpt_path)
        print(f'{ckpt_name}: PCE = {PCE:.4f}  Accuracy = {acc:.4f}  Macro F1 = {f1:.4f}')

        if PCE > best_pce:
            best_pce = PCE
            best_ckpt = ckpt_name
            best_acc = acc
            best_f1 = f1

    print(f'\nBest checkpoint: {best_ckpt}')
    print(f'  PCE:       {best_pce:.4f}')
    print(f'  Accuracy:  {best_acc:.4f}')
    print(f'  Macro F1:  {best_f1:.4f}')
    print(f'\nPaper baseline (MobileNetV2): PCE = 0.7610')
