import os

import torch
import torch.nn as nn

import models

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "assets", "models")


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(train_loader, model_name="cnn", epochs=25, lr=1e-3, device=None, verbose=True):
    # Trains a fresh model from scratch and returns it. Called once per LOSO fold
    # during evaluation, and once on all clips to produce the shipped model.
    # Any architecture registered in models.MODELS works here unchanged.
    device = device or pick_device()
    model = models.build(model_name).to(device)

    clips = train_loader.dataset.clips
    pos = sum(1 for c in clips if c.label == 1)
    neg = len(clips) - pos

    # Folds can be lopsided (one of yours trains on 943 tick vs 1336 noise).
    # pos_weight scales the positive class's share of the loss so the model can't
    # score well just by leaning toward whichever class happens to be commoner.
    pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x).squeeze(1)  # (batch, 1) -> (batch,), matching y
            loss = loss_fn(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            print(f"  epoch {epoch + 1:2d}/{epochs}  loss {total_loss / len(train_loader):.4f}")

    return model


def export_torchscript(model, clip_shape, name="cnn"):
    # TorchScript is what clipclass loads via torch::jit::load(). Trace on CPU in
    # eval mode — that bakes in inference behaviour for dropout and batch norm,
    # and matches how the C++ side runs.
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{name}.pt")

    model = model.to("cpu")
    model.eval()

    dummy = torch.randn(1, 1, *clip_shape)
    scripted = torch.jit.trace(model, dummy)
    scripted.save(path)

    return path
