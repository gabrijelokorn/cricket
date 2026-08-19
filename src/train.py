import os
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 1. DATASET ───────────────────────────────────────────────────────────────
# Reads .npy clips from assets/clips/noise/ (label=0) and assets/clips/courtship/ (label=1).
# Filenames look like <recName>_<start>_<end>_<clipMinFreq>_<clipMaxFreq>.npy.

# Recordings held out for validation — everything else goes to training.
# Pick recordings from a different night/session than anything else you're
# training on, not just a different file (adjacent-session recordings share
# background noise/weather and won't actually test generalization).
VAL_RECORDINGS = {
    "251204002733",
    "251204043041",
    "251204132716",
    "251204133853",
    "251204135029",
}

def parse_rec_name(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    return "_".join(parts[:-4])  # strips start, end, clip_min_freq, clip_max_freq


def load_samples(root="assets/clips"):
    raw = []  # (path, label, rec_name, shape)
    for label, folder in enumerate(["noise", "courtship"]):
        folder_path = os.path.join(root, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".npy"):
                path = os.path.join(folder_path, filename)
                shape = np.load(path, mmap_mode="r").shape
                raw.append((path, label, parse_rec_name(filename), shape))

    # A few clips near a recording's start/end can come out narrower than the
    # rest (padding to event_size ran off the edge of the file) — drop them
    # rather than let a mismatched shape crash batching later.
    expected_shape = Counter(shape for *_, shape in raw).most_common(1)[0][0]
    samples = [(p, l, rec) for p, l, rec, shape in raw if shape == expected_shape]

    skipped = len(raw) - len(samples)
    if skipped:
        print(f"Skipped {skipped} clip(s) with shape != {expected_shape} "
              f"(likely truncated near a recording boundary)")
    return samples


def print_stats(samples):
    # label 1 = positive (courtship), label 0 = negative (noise)
    per_rec = {}
    for _, label, rec in samples:
        counts = per_rec.setdefault(rec, {0: 0, 1: 0})
        counts[label] += 1

    total = {0: 0, 1: 0}
    for rec in sorted(per_rec):
        pos, neg = per_rec[rec][1], per_rec[rec][0]
        total[1] += pos
        total[0] += neg
        print(f"{rec}: positive: {pos}, negative: {neg}")

    print(f"TOTAL: positive: {total[1]}, negative: {total[0]}")


def split_by_recording(samples, val_recordings):
    val_recordings = set(val_recordings)
    all_recordings = set(rec for _, _, rec in samples)

    unknown = val_recordings - all_recordings
    if unknown:
        raise ValueError(f"VAL_RECORDINGS contains name(s) not found in the data: {sorted(unknown)}")

    train_samples = [(p, l) for p, l, rec in samples if rec not in val_recordings]
    val_samples   = [(p, l) for p, l, rec in samples if rec in val_recordings]

    if not val_samples:
        raise ValueError("VAL_RECORDINGS is empty — nothing to validate against.")
    if not train_samples:
        raise ValueError("VAL_RECORDINGS covers every recording — nothing left to train on.")

    def counts(subset):
        pos = sum(1 for _, l in subset if l == 1)
        neg = sum(1 for _, l in subset if l == 0)
        return pos, neg

    train_pos, train_neg = counts(train_samples)
    val_pos, val_neg = counts(val_samples)

    print(f"Split {len(all_recordings)} recording(s): "
          f"{len(all_recordings) - len(val_recordings)} train, {len(val_recordings)} valid")
    print(f"Validation recording(s): {', '.join(sorted(val_recordings))}")
    print(f"Train clips: {len(train_samples)} (positive: {train_pos}, negative: {train_neg})")
    print(f"Valid clips: {len(val_samples)} (positive: {val_pos}, negative: {val_neg})")
    return train_samples, val_samples


class SpectrogramDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples  # list of (path, label)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        arr = np.load(path).astype(np.float32) / 255.0  # match old ToTensor()'s [0,1] scale
        tensor = torch.from_numpy(arr).unsqueeze(0)      # (1, freq_bins, event_size)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ─── 2. MODEL ─────────────────────────────────────────────────────────────────
# Small CNN designed for your 300×24 (freq_bins × event_size) clips

class CricketCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: tall kernels to read frequency patterns
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),            # → ~8 × 150

            # Block 2
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),            # → ~4 × 75

            # Block 3
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 4)),    # → always 2 × 4, no matter what
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # 64 × 2 × 4 = 512 values
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.5),                 # prevents memorising the training set
            nn.Linear(64, 1),               # single score → sigmoid → 0 or 1
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── 3. TRAINING ──────────────────────────────────────────────────────────────

def train():
    # Load data and split by recording (not by clip — see split_by_recording)
    samples = load_samples("../assets/clips")
    print(f"Total clips: {len(samples)}")
    print_stats(samples)

    train_samples, val_samples = split_by_recording(samples, VAL_RECORDINGS)
    train_ds = SpectrogramDataset(train_samples)
    val_ds   = SpectrogramDataset(val_samples)
    val_size = len(val_ds)

    from torch.utils.data import WeightedRandomSampler
    weights = [2.0 if label == 1 else 1.0 for _, label in train_samples]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=16, sampler=sampler)  # remove shuffle=True
    val_loader   = DataLoader(val_ds,   batch_size=16, shuffle=False)

    # Model, loss, optimiser
    model     = CricketCNN()
    loss_fn   = nn.BCEWithLogitsLoss()      # works with raw scores (no sigmoid needed)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_acc = 0.0

    for epoch in range(25):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            predictions = model(images).squeeze(1)
            loss        = loss_fn(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                scores = model(images).squeeze(1)
                preds  = (torch.sigmoid(scores) > 0.5).float()
                correct += (preds == labels).sum().item()

        val_acc = correct / val_size

        print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.4f} | val accuracy: {val_acc:.1%}")

        # Save the best model seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "../assets/models/cricket.pth")
            print(f"           ✓ new best saved ({val_acc:.1%})")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.1%}")
    print("Model saved to assets/models/cricket.pth")

    # ── Export for C++ ────────────────────────────────────────────────
    model.load_state_dict(torch.load("../assets/models/cricket.pth"))  # load best weights
    model.eval()
    clip_shape = np.load(samples[0][0]).shape  # read real shape instead of hardcoding it
    dummy = torch.randn(1, 1, *clip_shape)
    scripted = torch.jit.trace(model, dummy)
    scripted.save("../assets/models/cricket.pt")
    print("Exported to assets/models/cricket.pt")

if __name__ == "__main__":
    train()