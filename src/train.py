import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 1. DATASET ───────────────────────────────────────────────────────────────
# Reads .npy clips from assets/clips/noise/ (label=0) and assets/clips/courtship/ (label=1).
# Filenames look like <recName>_<start>_<end>_<clipMinFreq>_<clipMaxFreq>.npy — the train/valid
# split groups by recName so clips from the same recording never end up on both sides.

def parse_rec_name(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split("_")
    return "_".join(parts[:-4])  # strips start, end, clip_min_freq, clip_max_freq


def load_samples(root="assets/clips"):
    samples = []  # (path, label, rec_name)
    for label, folder in enumerate(["noise", "courtship"]):
        folder_path = os.path.join(root, folder)
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".npy"):
                path = os.path.join(folder_path, filename)
                samples.append((path, label, parse_rec_name(filename)))
    return samples


def split_by_recording(samples, val_fraction=0.2, seed=42):
    recordings = sorted(set(rec for _, _, rec in samples))
    rng = random.Random(seed)
    rng.shuffle(recordings)

    if len(recordings) < 2:
        print(f"WARNING: only {len(recordings)} recording(s) found, can't do a recording-level "
              f"split — falling back to a random per-clip split (train/valid may leak).")
        items = list(samples)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_fraction))
        val_samples   = [(p, l) for p, l, _ in items[:n_val]]
        train_samples = [(p, l) for p, l, _ in items[n_val:]]
        return train_samples, val_samples

    n_val = max(1, int(len(recordings) * val_fraction))
    val_recs = set(recordings[:n_val])

    train_samples = [(p, l) for p, l, rec in samples if rec not in val_recs]
    val_samples   = [(p, l) for p, l, rec in samples if rec in val_recs]
    print(f"Split {len(recordings)} recording(s): "
          f"{len(recordings) - len(val_recs)} train, {len(val_recs)} valid")
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
# Small CNN designed for your 16×300 images

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
    samples = load_samples("assets/clips")
    print(f"Total clips: {len(samples)}")

    train_samples, val_samples = split_by_recording(samples)
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
            predictions = model(images).squeeze()
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
                scores = model(images).squeeze()
                preds  = (torch.sigmoid(scores) > 0.5).float()
                correct += (preds == labels).sum().item()

        val_acc = correct / val_size

        print(f"Epoch {epoch+1:2d} | loss: {avg_loss:.4f} | val accuracy: {val_acc:.1%}")

        # Save the best model seen so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "assets/models/cricket.pth")
            print(f"           ✓ new best saved ({val_acc:.1%})")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.1%}")
    print("Model saved to assets/models/cricket.pth")

    # ── Export for C++ ────────────────────────────────────────────────
    model.load_state_dict(torch.load("assets/models/cricket.pth"))  # load best weights
    model.eval()
    dummy = torch.randn(1, 1, 300, 16)
    scripted = torch.jit.trace(model, dummy)
    scripted.save("assets/models/cricket.pt")
    print("Exported to assets/models/cricket.pt")

if __name__ == "__main__":
    train()