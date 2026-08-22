import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "..", "..", "assets", "config.json")

# Same file the C++ side loads, so a value changed here takes effect in both
# training/evaluation and clipclass scanning — no second place to keep in sync.
with open(CONFIG_PATH) as f:
    CONFIG = json.load(f)

THRESHOLDS = CONFIG["thresholds"]
EVENT_SIZE = CONFIG["event_size"]
EVENT_STEP = CONFIG["event_step"]
WINDOW_SIZE = CONFIG["window_size"]
OVERLAP_SIZE = CONFIG["overlap_size"]
HOP_SIZE = WINDOW_SIZE - OVERLAP_SIZE
CLIP_MIN_FREQ = CONFIG["clip_min_freq"]
CLIP_MAX_FREQ = CONFIG["clip_max_freq"]
COURTSHIP_MIN_EVENTS = CONFIG["courtship_min_events"]
COURTSHIP_MAX_GAP = CONFIG["courtship_max_gap"]


def threshold_for(model_name):
    # Mirrors thresholdFor() in config.cpp — a missing entry is an error rather
    # than a silent default, since scanning with the wrong cutoff looks like it
    # worked but produces meaningless detections.
    if model_name not in THRESHOLDS:
        raise KeyError(f"no threshold configured for model '{model_name}' — "
                       f"add it to \"thresholds\" in assets/config.json")
    return THRESHOLDS[model_name]
