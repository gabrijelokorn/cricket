from loader import load_clips_by_date


def main():
    clips_by_date = load_clips_by_date()

    for date in sorted(clips_by_date):
        clips = clips_by_date[date]
        pos = sum(1 for c in clips if c.label == 1)
        neg = sum(1 for c in clips if c.label == 0)
        print(f"{date}: {len(clips)} clips (courtship: {pos}, noise: {neg})")


if __name__ == "__main__":
    main()
