from loader import load_clips_by_date, counts
from strategy import loso_folds


def main():
    clips_by_date = load_clips_by_date()

    print("Loaded clips per date:")
    for date in sorted(clips_by_date):
        pos, neg = counts(clips_by_date[date])
        print(f"  {date}: {pos + neg} clips (courtship: {pos}, noise: {neg})")

    folds = loso_folds(clips_by_date)
    print(f"\nLOSO: {len(folds)} fold(s)")

    for fold in folds:
        train_pos, train_neg = counts(fold.train)
        valid_pos, valid_neg = counts(fold.valid)
        print(f"\n  hold out {fold.held_out_date}")
        print(f"    train: {len(fold.train):4d} clips (courtship: {train_pos}, noise: {train_neg})")
        print(f"    valid: {len(fold.valid):4d} clips (courtship: {valid_pos}, noise: {valid_neg})")

        if not fold.train:
            print("    WARNING: nothing left to train on — need clips from more than one date")
        elif train_pos == 0 or valid_pos == 0:
            print("    WARNING: a side has no courtship clips — this fold can't measure anything useful")


if __name__ == "__main__":
    main()
