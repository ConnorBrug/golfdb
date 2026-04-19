import os.path as osp
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


_FRAME_CACHE = {}


def _preload_video(vid_dir, video_id):
    """Decode an entire video into a uint8 (N, H, W, 3) numpy array and cache it."""
    if video_id in _FRAME_CACHE:
        return _FRAME_CACHE[video_id]

    path = osp.join(vid_dir, f'{video_id}.mp4')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Could not open video: {path}')

    # preallocate from header frame count when available, write directly into the buffer
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 160
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 160

    if n_frames > 0:
        arr = np.empty((n_frames, h, w, 3), dtype=np.uint8)
        i = 0
        while i < n_frames:
            ret, img = cap.read()
            if not ret:
                break
            # write RGB directly into the preallocated slot, no intermediate alloc
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=arr[i])
            i += 1
        cap.release()
        if i == 0:
            raise RuntimeError(f'Video has 0 frames: {path}')
        if i < n_frames:
            arr = arr[:i].copy()
    else:
        frames = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f'Video has 0 frames: {path}')
        arr = np.stack(frames, axis=0)

    _FRAME_CACHE[video_id] = arr
    return arr


def preload_all_videos(vid_dir, video_ids, verbose=True):
    from tqdm import tqdm
    iterator = tqdm(video_ids, desc='Preloading videos', unit='vid') if verbose else video_ids
    total_bytes = 0
    for vid in iterator:
        arr = _preload_video(vid_dir, vid)
        total_bytes += arr.nbytes
    if verbose:
        print(f'Cached {len(video_ids)} videos, {total_bytes / 1e9:.2f} GB in RAM')


class GolfDB(Dataset):
    """Workers only slice frames. All augmentation happens on GPU in train.py."""

    def __init__(self, data_file, vid_dir, seq_length, train=True, preload=True):
        self.df = pd.read_pickle(data_file)
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.train = train

        if preload:
            preload_all_videos(vid_dir, self.df['id'].tolist(), verbose=True)

    def __len__(self):
        return len(self.df)

    def _get_video(self, video_id):
        if video_id in _FRAME_CACHE:
            return _FRAME_CACHE[video_id]
        return _preload_video(self.vid_dir, video_id)

    def __getitem__(self, idx):
        a = self.df.loc[idx, :]
        events = np.asarray(a['events'], dtype=np.int64)
        events = events - events[0]
        inner_events = events[1:-1]  # the 8 swing events we actually classify

        frames = self._get_video(a['id'])
        total_frames = frames.shape[0]

        if self.train:
            # random start up to the last event, then walk forward seq_length frames wrapping at end
            start_frame = np.random.randint(events[-1] + 1)
            positions = (np.arange(self.seq_length, dtype=np.int64) + start_frame) % total_frames

            # vectorized gather, no Python per-frame loop
            images = frames[positions]

            labels = np.full(self.seq_length, 8, dtype=np.int64)
            for evt_idx, evt_pos in enumerate(inner_events):
                labels[positions == evt_pos] = evt_idx
        else:
            # full-length eval: label defaults to no-event (8), stamp in the 8 event frames
            images = frames
            labels = np.full(total_frames, 8, dtype=np.int64)
            for evt_idx, evt_pos in enumerate(inner_events):
                if 0 <= evt_pos < total_frames:
                    labels[evt_pos] = evt_idx

        # CHW uint8 contiguous for a fast pinned host-device copy
        images = np.ascontiguousarray(images.transpose((0, 3, 1, 2)))
        return {
            'images': torch.from_numpy(images),
            'labels': torch.from_numpy(labels),
        }
