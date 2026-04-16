import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_preds(probs, labels, tol=-1):
    """
    Gets correct events in full-length sequence using tolerance based on number of frames from address to impact.
    Used during validation only.
    :param probs: (sequence_length, 9)
    :param labels: (sequence_length,)
    :return: array indicating correct events in predicted sequence (8,)
    """

    events = np.where(labels < 8)[0]
    preds = np.zeros(len(events))
    if tol == -1:
        tol = int(max(np.round((events[5] - events[0])/30), 1))
    for i in range(len(events)):
        preds[i] = np.argsort(probs[:, i])[-1]
    deltas = np.abs(events-preds)
    correct = (deltas <= tol).astype(np.uint8)
    return events, preds, deltas, tol, correct


def freeze_layers(num_freeze, net):
    """Freeze the first num_freeze layers of the MobileNetV3 backbone.

    Layers are counted as: conv_stem, bn1, then each individual
    InvertedResidual block flattened from the blocks Sequential.
    This gives ~18 freezable layers, similar to MobileNetV2's structure.
    Freezing 10 (the paper's default) covers the stem + first 8 blocks.
    """
    import torch.nn as nn
    cnn = next(net.children())  # self.cnn (timm MobileNetV3 model)

    # Build flat list: stem layers, then individual inverted-residual blocks
    freezable = [cnn.conv_stem, cnn.bn1]
    for stage in cnn.blocks:
        if isinstance(stage, nn.Sequential):
            for block in stage:
                freezable.append(block)
        else:
            freezable.append(stage)

    num_to_freeze = min(num_freeze, len(freezable))
    for i in range(num_to_freeze):
        for param in freezable[i].parameters():
            param.requires_grad = False
    print(f"Froze {num_to_freeze}/{len(freezable)} backbone layers")