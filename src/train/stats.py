import argparse

from loader import load_clips_by_date, counts, parse_date

# Read-only clip tally. Groups by date (default) or by recording, so you can see
# how labeling is progressing without running training or evaluation.


def group_by_recording(clips_by_date):
    by_rec = {}
    for clips in clips_by_date.values():
        for c in clips:
            by_rec.setdefault(c.rec_name, []).append(c)
    return by_rec


def print_group(title, groups):
    print(f"{title:<24} {'ticks':>10} {'noise':>8} {'total':>8}")
    print("-" * 52)

    tot_pos = tot_neg = 0
    for key in sorted(groups):
        pos, neg = counts(groups[key])
        tot_pos += pos
        tot_neg += neg
        print(f"{key:<24} {pos:>10} {neg:>8} {pos + neg:>8}")

    print("-" * 52)
    print(f"{'TOTAL':<24} {tot_pos:>10} {tot_neg:>8} {tot_pos + tot_neg:>8}")
    print(f"({len(groups)} groups)")


def main(by="date"):
    clips_by_date = load_clips_by_date()

    if by == "recording":
        print_group("recording", group_by_recording(clips_by_date))
    else:
        print_group("date", clips_by_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--by", default="date", choices=["date", "recording"],
                        help="group clip counts by date (default) or by recording")
    main(parser.parse_args().by)
