import os
import csv
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from loader import load_clips_by_date, counts
from strategy import loso_folds
from dataset import ClipDataset
from trainer import train_model, pick_device
from models import MODELS
from config import threshold_for

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "output", "evaluation")

EPOCHS = 25
BATCH_SIZE = 16

METRIC_NAMES = ["accuracy", "balanced_accuracy", "precision", "recall", "f1"]


def predict_scores(model, clips, batch_size=32, device=None):
    # Returns (y_true, scores) — scores are sigmoid outputs in [0,1], the same
    # numbers pipeline.cpp thresholds at inference time.
    device = device or pick_device()
    model = model.to(device).eval()

    loader = DataLoader(ClipDataset(clips), batch_size=batch_size, shuffle=False)

    y_true, scores = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device)).squeeze(1)
            scores.append(torch.sigmoid(logits).cpu().numpy())
            y_true.append(y.numpy())

    return np.concatenate(y_true), np.concatenate(scores)


def confusion(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def metrics_from_confusion(tn, fp, fn, tp):
    # Every metric below is just a different ratio of these same four counts.
    total = tn + fp + fn + tp

    accuracy = (tp + tn) / total if total else 0.0

    recall = tp / (tp + fn) if (tp + fn) else 0.0        # of real courtships, how many found
    specificity = tn / (tn + fp) if (tn + fp) else 0.0   # of real noise, how much correctly ignored
    balanced_accuracy = (recall + specificity) / 2       # accuracy's baseline moves with class
                                                         # balance; this one always baselines at 0.5

    precision = tp / (tp + fp) if (tp + fp) else 0.0     # of flagged clips, how many were real
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_predictions(y_true, y_pred):
    tn, fp, fn, tp = confusion(y_true, y_pred)
    return metrics_from_confusion(tn, fp, fn, tp), (tn, fp, fn, tp)


def mean_row(metric_rows):
    # Averaging across folds is the number worth quoting — a single fold's score
    # depends heavily on which date landed in validation.
    mean = {"fold": "MEAN"}
    for metric in METRIC_NAMES:
        mean[metric] = sum(r[metric] for r in metric_rows) / len(metric_rows)
    return mean


def run_loso_evaluation(model_name="cnn", epochs=EPOCHS, threshold=None):
    clips_by_date = load_clips_by_date()
    folds = loso_folds(clips_by_date)

    if any(not f.train for f in folds):
        raise ValueError("some fold has no training data — need clips from more than one date")

    # Each model gets its own cutoff — scores from different architectures aren't
    # on the same scale, so one shared value would mean different things.
    if threshold is None:
        threshold = threshold_for(model_name)
    print(f"threshold: {threshold} (thresholds.{model_name} in assets/config.json)")

    metric_rows, confusions = [], []

    for fold in folds:
        print(f"\n=== fold: hold out {fold.held_out_date} ===")
        train_pos, train_neg = counts(fold.train)
        valid_pos, valid_neg = counts(fold.valid)
        print(f"train {len(fold.train)} (pos {train_pos}, neg {train_neg}) | "
              f"valid {len(fold.valid)} (pos {valid_pos}, neg {valid_neg})")

        train_loader = DataLoader(ClipDataset(fold.train), batch_size=BATCH_SIZE, shuffle=True)
        model = train_model(train_loader, model_name=model_name, epochs=epochs)

        y_true, scores = predict_scores(model, fold.valid)
        y_pred = (scores > threshold).astype(float)

        metrics, cm = evaluate_predictions(y_true, y_pred)
        metric_rows.append({"fold": fold.held_out_date, **metrics})
        confusions.append((fold.held_out_date, cm))

        print("  " + "  ".join(f"{k}={metrics[k]:.3f}" for k in METRIC_NAMES))

    metric_rows.append(mean_row(metric_rows))
    return metric_rows, confusions


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"wrote {path}")


def write_confusion_matrix(cm, path):
    # Laid out the conventional way: rows are what the clip actually was,
    # columns are what the model said.
    tn, fp, fn, tp = cm

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["", "predicted_noise", "predicted_courtship"])
        writer.writerow(["actual_noise", tn, fp])
        writer.writerow(["actual_courtship", fn, tp])
    print(f"wrote {path}")


def evaluate_one(model_name):
    metric_rows, confusions = run_loso_evaluation(model_name=model_name)

    # Results live under the model's own name so comparing architectures doesn't
    # silently overwrite the previous run's numbers.
    out_dir = os.path.join(RESULTS_DIR, model_name)

    print()
    write_csv(metric_rows, os.path.join(out_dir, "metrics.csv"),
              ["fold"] + METRIC_NAMES)

    for fold_name, cm in confusions:
        write_confusion_matrix(cm, os.path.join(out_dir, f"{fold_name}_confusionmatrix.csv"))


def main(model_name="cnn"):
    # Note: these are architectures from models.MODELS, not the .pt files in
    # assets/models. Each one is trained from scratch per fold — a saved .pt was
    # trained on every clip, so there'd be no honest data left to score it on.
    names = sorted(MODELS) if model_name == "all" else [model_name]

    for name in names:
        print(f"\n########## {name} ##########")
        evaluate_one(name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=sorted(MODELS) + ["all"],
                        help="architecture to evaluate, or 'all' to evaluate every registered one")
    main(parser.parse_args().model)
