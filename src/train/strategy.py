from dataclasses import dataclass
from typing import List

from loader import Clip


@dataclass
class Fold:
    held_out_date: str
    train: List[Clip]
    valid: List[Clip]


def loso_folds(clips_by_date):
    # Leave One Session Out: one fold per date, that date's clips validate,
    # every other date trains. Splitting by date (not by clip) is what keeps
    # near-duplicate clips from the same session out of both sides.
    dates = sorted(clips_by_date)

    folds = []
    for held_out in dates:
        valid = clips_by_date[held_out]
        train = [c for d in dates if d != held_out for c in clips_by_date[d]]
        folds.append(Fold(held_out_date=held_out, train=train, valid=valid))

    return folds
