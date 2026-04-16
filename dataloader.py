import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def _open_video(self, video_id):
        path = osp.join(self.vid_dir, f'{video_id}.mp4')
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f'Could not open video: {path}')
        return cap, path

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]
        events = a['events'].copy()
        events -= events[0]  # align event frames to the trimmed clip

        cap, path = self._open_video(a['id'])
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(f'Video has 0 frames or unreadable metadata: {path}')

        if self.train:
            images = np.empty((self.seq_length, 160, 160, 3), dtype=np.uint8)
            labels = np.empty((self.seq_length,), dtype=np.int64)

            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            n = 0

            while n < self.seq_length:
                ret, img = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[n] = img

                if pos in events[1:-1]:
                    labels[n] = np.where(events[1:-1] == pos)[0][0]
                else:
                    labels[n] = 8

                n += 1
                pos += 1

            cap.release()
        else:
            images = np.empty((total_frames, 160, 160, 3), dtype=np.uint8)
            labels = np.empty((total_frames,), dtype=np.int64)

            for pos in range(total_frames):
                ret, img = cap.read()
                if not ret:
                    cap.release()
                    raise RuntimeError(f'Failed reading frame {pos} from {path}')
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images[pos] = img
                if pos in events[1:-1]:
                    labels[pos] = np.where(events[1:-1] == pos)[0][0]
                else:
                    labels[pos] = 8

            cap.release()

        sample = {'images': images, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


# -----------------------------
# Sequence-consistent transforms
# -----------------------------

class RandomHorizontalFlip:
    """Flip all frames in a sequence together."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample['images'] = sample['images'][:, :, ::-1, :].copy()
        return sample


class ColorJitter:
    """Apply the same color jitter to every frame in the sequence."""
    def __init__(self, brightness=0.20, contrast=0.20, saturation=0.15):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample):
        images = sample['images'].astype(np.float32)

        if self.brightness > 0:
            b = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            images *= b

        if self.contrast > 0:
            c = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean = images.mean(axis=(1, 2, 3), keepdims=True)
            images = (images - mean) * c + mean

        if self.saturation > 0:
            s = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            gray = np.mean(images, axis=-1, keepdims=True)
            images = gray + (images - gray) * s

        sample['images'] = np.clip(images, 0, 255).astype(np.uint8)
        return sample


class RandomRotation:
    """Rotate every frame in a sequence by the same small angle."""
    def __init__(self, degrees=5):
        self.degrees = degrees

    def __call__(self, sample):
        angle = np.random.uniform(-self.degrees, self.degrees)
        if abs(angle) < 0.25:
            return sample

        images = sample['images']
        h, w = images.shape[1], images.shape[2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        sample['images'] = np.stack(
            [cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
             for img in images],
            axis=0
        )
        return sample


# -----------------------------
# Standard transforms
# -----------------------------

class ToTensor:
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {
            'images': torch.from_numpy(images).float().div(255.0),
            'labels': torch.from_numpy(labels).long(),
        }


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}
