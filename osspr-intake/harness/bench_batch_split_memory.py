"""Peak memory of SubtitleBatcher._split_lines on one long evenly timed scene.

Auto-captions often share one frame gap, so no gap exceeds the 60s scene
threshold and the whole file is one scene. Equal gaps make the earliest
maximum win, and each recursive call peels min_batch_size lines while every
suffix list stays live. Defaults are min_batch_size 10 and max_batch_size 100.

8000 lines at 2.0s plus an 83ms gap is about 4.6 hours. That is under the
recursion limit (about 10000 lines crashes). The lines are built before
tracing, so the peak is the split itself.

Noise is (max - min) / median * 100 over five traced runs after one warmup.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from PySubtrans.SettingsType import SettingsType
from PySubtrans.SubtitleBatcher import SubtitleBatcher
from PySubtrans.SubtitleLine import SubtitleLine

from common import fmt_bytes, measure_peak_bytes

LINE_COUNT = 8000
MIN_BATCH = 10
MAX_BATCH = 100


def build_lines() -> list[SubtitleLine]:
    """Evenly timed captions. Every gap is two frames at 24fps, so the max is tied."""
    lines : list[SubtitleLine] = []
    cursor = timedelta(0)
    gap = timedelta(milliseconds=83)
    duration = timedelta(milliseconds=2000)
    for index in range(LINE_COUNT):
        end = cursor + duration
        lines.append(SubtitleLine.Construct(
            index + 1, cursor, end, f"Caption {index} stays inside one long scene."
        ))
        cursor = end + gap
    return lines


def main() -> None:
    lines = build_lines()
    batcher = SubtitleBatcher(SettingsType({
        "min_batch_size": MIN_BATCH,
        "max_batch_size": MAX_BATCH,
        "scene_threshold": 60.0,
        "prevent_overlapping_times": False,
    }))

    def split() -> list[list[SubtitleLine]]:
        return batcher._split_lines(lines)

    samples, median, noise = measure_peak_bytes(split)
    groups = split()
    expected_groups = (LINE_COUNT - MAX_BATCH) // MIN_BATCH + 1
    if len(groups) != expected_groups or len(groups[0]) != MIN_BATCH or len(groups[-1]) != MAX_BATCH:
        raise SystemExit(
            f"unexpected split shape groups={len(groups)} first={len(groups[0])} last={len(groups[-1])}"
        )
    flat = [line.number for group in groups for line in group]
    if flat != list(range(1, LINE_COUNT + 1)):
        raise SystemExit("split did not keep every line once, in order")

    digest = hashlib.sha256(",".join(str(len(group)) for group in groups).encode("utf-8")).hexdigest()[:16]
    print(fmt_bytes(samples, median, noise))
    print(f"groups={len(groups)}")
    print(f"checksum={digest}")


if __name__ == "__main__":
    main()
