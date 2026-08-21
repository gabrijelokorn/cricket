import os
from dataclasses import dataclass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLIPS_ROOT = os.path.join(SCRIPT_DIR, "..", "..", "assets", "clips")


@dataclass
class Clip:
    path: str
    label: int  # 1 = courtship, 0 = noise
    rec_name: str


def parse_rec_name(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.split(".")
    return ".".join(parts[:-2])  # strips the <startMs>_<endMs> and <minFreq>_<maxFreq> groups


def parse_date(rec_name):
    # e.g. "raven.251202_125905" -> "251202"
    return rec_name.split(".")[1].split("_")[0]


def load_clips_by_date(root=CLIPS_ROOT):
    clips_by_date = {}

    for label, folder in enumerate(["noise", "courtship"]):
        folder_path = os.path.join(root, folder)
        for filename in os.listdir(folder_path):
            if not filename.lower().endswith(".npy"):
                continue

            rec_name = parse_rec_name(filename)
            date = parse_date(rec_name)
            clip = Clip(path=os.path.join(folder_path, filename), label=label, rec_name=rec_name)

            clips_by_date.setdefault(date, []).append(clip)

    return clips_by_date
