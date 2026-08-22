import argparse

from loader import load_clips_by_date, counts
from strategy import loso_folds
from dataset import ClipDataset, clip_shape, check_shapes
from trainer import train_model, export_torchscript
from models import MODELS

from torch.utils.data import DataLoader

EPOCHS = 25
BATCH_SIZE = 16


def main(model_name="cnn"):
    clips_by_date = load_clips_by_date()
    all_clips = [c for date in sorted(clips_by_date) for c in clips_by_date[date]]

    print("Loaded clips per date:")
    for date in sorted(clips_by_date):
        pos, neg = counts(clips_by_date[date])
        print(f"  {date}: {pos + neg} clips (courtship: {pos}, noise: {neg})")

    shapes = check_shapes(all_clips)
    if len(shapes) > 1:
        print("\nERROR: clips have inconsistent shapes — regenerate them under one config:")
        for shape, clips in shapes.items():
            print(f"  {shape}: {len(clips)} clip(s)")
        return

    shape = clip_shape(all_clips)
    print(f"\nClip shape: {shape}")

    folds = loso_folds(clips_by_date)
    print(f"LOSO: {len(folds)} fold(s) — used for evaluation, not yet wired up")

    # The shipped model trains on everything. Cross-validation measures how well
    # this setup generalises; it doesn't produce the model you deploy — holding a
    # date out of the final model would just throw away data for no benefit.
    pos, neg = counts(all_clips)
    print(f"\nTraining on all {len(all_clips)} clips (courtship: {pos}, noise: {neg})")

    loader = DataLoader(ClipDataset(all_clips), batch_size=BATCH_SIZE, shuffle=True)
    model = train_model(loader, model_name=model_name, epochs=EPOCHS)

    path = export_torchscript(model, shape, name=model_name)
    print(f"\nExported to {path}")
    print("NOTE: no honest accuracy figure yet — this model saw every clip during "
          "training. LOSO evaluation is the next step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="cnn", choices=sorted(MODELS),
                        help="architecture to train; also the exported filename")
    main(parser.parse_args().model)
