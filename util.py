"""Small utilities: running-average meter, PCE metric from the paper,
and a helper to freeze N early blocks of the timm MobileNetV3 backbone."""

import numpy as np


class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def correct_preds(probs, labels, tol=-1):
    """
    Gets correct events in full-length sequence using tolerance based on number of frames from address to impact.
    Used during validation only.

    probs: (sequence_length, 9)
    labels: (sequence_length,)
    returns: events, preds, deltas, tol, correct
    """
    events = np.where(labels < 8)[0]
    preds = np.zeros(len(events))

    if tol == -1:
        tol = int(max(np.round((events[5] - events[0]) / 30), 1))

    for i in range(len(events)):
        preds[i] = np.argsort(probs[:, i])[-1]

    deltas = np.abs(events - preds)
    correct = (deltas <= tol).astype(np.uint8)
    return events, preds, deltas, tol, correct


def freeze_layers(num_freeze, net):
    """
    Freeze the first num_freeze logical layers of the MobileNetV3 backbone.

    Layer order:
      1. conv_stem
      2. bn1
      3+. each InvertedResidual block flattened from cnn.blocks

    MobileNetV3-Large in timm ends up with ~18 freezable chunks here.
    """
    import torch.nn as nn

    cnn = net.cnn
    freezable = [cnn.conv_stem, cnn.bn1]

    for stage in cnn.blocks:
        if isinstance(stage, nn.Sequential):
            for block in stage:
                freezable.append(block)
        else:
            freezable.append(stage)

    num_to_freeze = min(max(int(num_freeze), 0), len(freezable))
    for i in range(num_to_freeze):
        for param in freezable[i].parameters():
            param.requires_grad = False

    print(f'Froze {num_to_freeze}/{len(freezable)} backbone layers')
