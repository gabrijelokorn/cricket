import torch.nn as nn


class CricketCNN(nn.Module):
    # Input is (1, freq_bins, event_size) — currently (1, 300, 24): a tall, narrow
    # image, 300 frequency bins by 24 time frames.

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Kernels are taller than they are wide (5 in frequency, 3 in time):
            # a chirp is a narrow horizontal band, so frequency context is what
            # distinguishes it from broadband noise.
            nn.Conv2d(1, 16, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 300x24 -> 150x12

            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),  # 150x12 -> 75x6

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Always emits 2x4 regardless of input size, so the classifier below
            # keeps a fixed 512 inputs even if config.json changes clip dimensions.
            # The flip side: a wrong-sized clip runs silently instead of erroring,
            # which is why dataset.check_shapes() exists.
            nn.AdaptiveAvgPool2d((2, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 64 * 2 * 4 = 512
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.5),  # keeps the net from memorising individual clips
            nn.Linear(64, 1),  # raw logit — sigmoid is applied by the loss, and by C++ at inference
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# Every architecture worth trying gets registered here. The key doubles as the
# exported filename (assets/models/<key>.pt) and as clipclass's --model value.
MODELS = {
    "cnn": CricketCNN,
}


def build(name):
    if name not in MODELS:
        raise ValueError(f"unknown model '{name}' — available: {', '.join(sorted(MODELS))}")
    return MODELS[name]()
