"""Single-video demo for the trained SwingNet model.

Loads one .mp4 file, resizes/pads to 160x160, runs the model, and prints the
predicted frame index for each of the 8 swing events plus the confidence at that
frame. Pass --show to also display the annotated frames.

Example:
  python test_video.py -p test_video.mp4 \
      --ckpt models_s1/swingnet_ema_8000.pth.tar
"""

import argparse
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

from gpu_augment import augment_and_normalize
from model import EventDetector


EVENT_NAMES = {
    0: 'Address',
    1: 'Toe-up',
    2: 'Mid-backswing (arm parallel)',
    3: 'Top',
    4: 'Mid-downswing (arm parallel)',
    5: 'Impact',
    6: 'Mid-follow-through (shaft parallel)',
    7: 'Finish',
}


def load_video(path, input_size=160):
    """Read a video, resize to fit within input_size while preserving aspect ratio,
    and pad the remainder with the ImageNet mean color. Returns (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {path}')

    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ratio = input_size / max(h, w)
    new_h, new_w = int(h * ratio), int(w * ratio)
    pad_h, pad_w = input_size - new_h, input_size - new_w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    # ImageNet mean in BGR for cv2.copyMakeBorder
    pad_color = [0.406 * 255, 0.456 * 255, 0.485 * 255]

    frames = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        resized = cv2.resize(img, (new_w, new_h))
        padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=pad_color)
        frames.append(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))
    cap.release()

    return np.stack(frames, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', default='test_video.mp4', help='Path to .mp4')
    parser.add_argument('-s', '--seq-length', type=int, default=64,
                        help='Frames per forward pass')
    parser.add_argument('--ckpt', required=True,
                        help='Checkpoint file, e.g. models_s1/swingnet_ema_8000.pth.tar')
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--lstm-hidden', type=int, default=256)
    parser.add_argument('--show', action='store_true', help='Display annotated event frames')
    parser.add_argument('--fp16', action='store_true')
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f'Checkpoint not found: {args.ckpt}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    amp_dtype = torch.float16 if args.fp16 else torch.bfloat16

    print(f'Loading video: {args.path}')
    frames = load_video(args.path)  # (T, H, W, 3) uint8
    # to (1, T, 3, H, W) uint8 for the GPU augment pipeline
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).contiguous()

    print(f'Loading checkpoint: {args.ckpt}')
    model = EventDetector(
        pretrain=False,
        width_mult=1.0,
        lstm_layers=args.lstm_layers,
        lstm_hidden=args.lstm_hidden,
        bidirectional=True,
        dropout=False,
    )
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device).to(memory_format=torch.channels_last)
    model.eval()

    # inference in seq_length chunks so long clips fit in VRAM
    total = video.shape[1]
    all_probs = []
    with torch.no_grad():
        batch = 0
        while batch * args.seq_length < total:
            start = batch * args.seq_length
            end = min((batch + 1) * args.seq_length, total)
            chunk_u8 = video[:, start:end].to(device, non_blocking=True)
            with autocast('cuda', dtype=amp_dtype):
                chunk = augment_and_normalize(chunk_u8, train=False, dtype=amp_dtype)
                logits = model(chunk)
            all_probs.append(F.softmax(logits.float(), dim=1).cpu().numpy())
            batch += 1
    probs = np.concatenate(all_probs, axis=0)  # (T, 9)

    # take argmax frame for each of the 8 event columns
    event_frames = np.argmax(probs[:, :8], axis=0)
    confidences = [float(probs[f, i]) for i, f in enumerate(event_frames)]

    print('\nPredicted event frames:')
    for i, (f, c) in enumerate(zip(event_frames, confidences)):
        print(f'  {EVENT_NAMES[i]:<40} frame {f:4d}   confidence {c:.3f}')

    if args.show:
        cap = cv2.VideoCapture(args.path)
        for i, frame_idx in enumerate(event_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ret, img = cap.read()
            if not ret:
                continue
            label = f'{EVENT_NAMES[i]}: {confidences[i]:.3f}'
            cv2.putText(img, label, (10, 25), cv2.FONT_HERSHEY_DUPLEX,
                        0.75, (0, 0, 255), 2)
            cv2.imshow(EVENT_NAMES[i], img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()


if __name__ == '__main__':
    main()
