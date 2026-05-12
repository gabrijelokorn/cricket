import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# ─── 1. DATASET ───────────────────────────────────────────────────────────────
# Reads images from data/courting/ (label=1) and data/noise/ (label=0)

class SpectrogramDataset(Dataset):
    def __init__(self, root="data"):
        self.samples = []
        for label, folder in enumerate(["noise", "courtship"]):
            folder_path = os.path.join(root, folder)
            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".png", ".jpg")):
                    self.samples.append((os.path.join(folder_path, filename), label))

        self.transform = transforms.Compose([
            transforms.Grayscale(),          # make sure it's 1 channel
            transforms.ToTensor(),           # converts to [0.0, 1.0]
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = self.transform(Image.open(path))
        return img, torch.tensor(label, dtype=torch.float32)


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
    # Load data and split
    dataset = SpectrogramDataset("assets/clips")
    print(f"Total images: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    from torch.utils.data import WeightedRandomSampler
    weights = []
    for _, label in train_ds:
        weights.append(2.0 if label == 1.0 else 1.0)
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
            torch.save(model.state_dict(), "cricket_model.pth")
            print(f"           ✓ new best saved ({val_acc:.1%})")

    print(f"\nDone. Best validation accuracy: {best_val_acc:.1%}")
    print("Model saved to cricket_model.pth")


if __name__ == "__main__":
    train()