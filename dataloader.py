import os.path as osp
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class GolfDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]  # annotation info
        events = a['events']
        events -= events[0]  # now frame #s correspond to frames in preprocessed video clips

        images, labels = [], []
        cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

        if self.train:
            # random starting position, sample 'seq_length' frames
            start_frame = np.random.randint(events[-1] + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            pos = start_frame
            while len(images) < self.seq_length:
                ret, img = cap.read()
                if ret:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(img)
                    if pos in events[1:-1]:
                        labels.append(np.where(events[1:-1] == pos)[0][0])
                    else:
                        labels.append(8)
                    pos += 1
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    pos = 0
            cap.release()
        else:
            # full clip
            for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                _, img = cap.read()
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images.append(img)
                if pos in events[1:-1]:
                    labels.append(np.where(events[1:-1] == pos)[0][0])
                else:
                    labels.append(8)
            cap.release()

        sample = {'images': np.asarray(images), 'labels': np.asarray(labels)}
        if self.transform:
            sample = self.transform(sample)
        return sample


# ── Data augmentation transforms (applied consistently across all frames in a sequence) ──

class RandomHorizontalFlip(object):
    """Randomly flip all frames horizontally with probability p.
    Golf swings are symmetric (left/right handed) so flipping is valid."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if np.random.random() < self.p:
            sample['images'] = sample['images'][:, :, ::-1, :].copy()
        return sample


class ColorJitter(object):
    """Apply random brightness, contrast, and saturation jitter.
    Same random factors applied to every frame in the sequence."""
    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample):
        images = sample['images'].astype(np.float32)

        # brightness
        if self.brightness > 0:
            factor = 1.0 + np.random.uniform(-self.brightness, self.brightness)
            images = images * factor

        # contrast
        if self.contrast > 0:
            factor = 1.0 + np.random.uniform(-self.contrast, self.contrast)
            mean = images.mean()
            images = (images - mean) * factor + mean

        # saturation
        if self.saturation > 0:
            factor = 1.0 + np.random.uniform(-self.saturation, self.saturation)
            gray = np.mean(images, axis=-1, keepdims=True)
            images = gray + (images - gray) * factor

        sample['images'] = np.clip(images, 0, 255).astype(np.uint8)
        return sample


class RandomRotation(object):
    """Apply small random rotation consistently to all frames.
    Handles varying camera angles in the dataset."""
    def __init__(self, degrees=5):
        self.degrees = degrees

    def __call__(self, sample):
        angle = np.random.uniform(-self.degrees, self.degrees)
        if abs(angle) < 0.5:
            return sample  # skip trivial rotations
        images = sample['images']
        h, w = images.shape[1], images.shape[2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        sample['images'] = np.stack([cv2.warpAffine(img, M, (w, h)) for img in images])
        return sample


# ── Standard transforms ──

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    dataset = GolfDB(data_file='data/train_split_1.pkl',
                     vid_dir='data/videos_160/',
                     seq_length=64,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 8)[0]
        print('{} events: {}'.format(len(events), events))
