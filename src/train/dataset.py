import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ClipDataset(Dataset):
    def __init__(self, clips):
        self.clips = clips  # list of loader.Clip

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        clip = self.clips[i]

        # Clips are stored as float spectrograms scaled to [0,255] by Wav::getSpec().
        # Dividing by 255 puts them in [0,1] — the C++ side does the same before
        # inference (pipeline.cpp scoreWindow), so training and scanning agree.
        arr = np.load(clip.path).astype(np.float32) / 255.0

        x = torch.from_numpy(arr).unsqueeze(0)  # (1, freq_bins, event_size) — add channel dim
        y = torch.tensor(clip.label, dtype=torch.float32)
        return x, y


def clip_shape(clips):
    # Reads one clip to learn (freq_bins, event_size) instead of hardcoding it,
    # so changing window_size/clip freq range in config.json doesn't silently break things.
    return np.load(clips[0].path).shape


def check_shapes(clips):
    # A clip whose spectrogram is a different size than the rest can't be batched
    # with them (torch.stack throws). Usually means clips were generated under
    # different config.json settings, or got truncated at a recording boundary.
    shapes = {}
    for c in clips:
        shape = np.load(c.path, mmap_mode="r").shape
        shapes.setdefault(shape, []).append(c)
    return shapes


def make_loaders(train_clips, valid_clips, batch_size=16):
    train_loader = DataLoader(ClipDataset(train_clips), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(ClipDataset(valid_clips), batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader
