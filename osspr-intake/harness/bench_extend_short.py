"""Time Subtitles._extend_short_subtitles on one dense translated file.

extend_short_subtitles is off unless the user saves with it enabled.
This is that save: 8000 cues over a little under five hours of start times,
each 0.4s long and 1.6s before the next, the default 0.1s per character.
The last cue is the grapheme cluster the unit test already counts as three.
"""
import gc
import logging
import statistics
import sys
import time
from datetime import timedelta

from PySubtrans.SettingsType import SettingsType
from PySubtrans.SubtitleLine import SubtitleLine
from PySubtrans.Subtitles import SaveSettings, Subtitles

logging.disable(logging.CRITICAL)

CUE_COUNT = 8000
RUNS = 7
PLAIN = "She waited by the door until the rain stopped."
TAGGED = "<i>Wait, what did you say to her?</i>"
CJK = "これはテストです。本当に雨が降っています。"
EMOJI = "A <i>好</i>\n👨‍👩‍👧‍👦"
TEXTS = (PLAIN, TAGGED, CJK)


def build_lines() -> list[SubtitleLine]:
    lines : list[SubtitleLine] = []
    for index in range(CUE_COUNT - 1):
        start = timedelta(seconds=index * 2)
        lines.append(SubtitleLine.Construct(index + 1, start, start + timedelta(milliseconds=400), TEXTS[index % 3]))
    start = timedelta(seconds=(CUE_COUNT - 1) * 2)
    lines.append(SubtitleLine.Construct(CUE_COUNT, start, start + timedelta(milliseconds=400), EMOJI))
    return lines


def main() -> None:
    lines = build_lines()
    original_ends = [line.end for line in lines]
    save_settings = SaveSettings(SettingsType({
        'min_line_duration': 0.8,
        'seconds_per_character': 0.1,
        'min_gap': 0.05,
    }))
    subtitles = Subtitles()

    def once() -> list[SubtitleLine]:
        return subtitles._extend_short_subtitles(lines, save_settings)

    adjusted = once()
    # 37 visible letters at 0.1s is 3.7s, which the next cue caps at 1.95s.
    capped = lines[0].start + timedelta(seconds=1.95)
    # Three graphemes are below the 0.8s minimum, and nothing follows this cue.
    minimum = lines[-1].start + timedelta(seconds=0.8)
    if len(adjusted) != CUE_COUNT or adjusted[0].end != capped or adjusted[-1].end != minimum:
        sys.exit("extended durations did not match the reading-time rules")
    if [line.end for line in lines] != original_ends:
        sys.exit("extend_short_subtitles modified the stored lines")

    samples : list[float] = []
    for _run in range(RUNS):
        gc.collect()
        started = time.perf_counter()
        once()
        samples.append((time.perf_counter() - started) * 1000.0)

    median = statistics.median(samples)
    noise = (max(samples) - min(samples)) / median * 100.0
    print(f"median_ms={median:.3f}")
    print(f"noise_pct={noise:.4f}")
    print("runs_ms=" + ",".join(f"{sample:.3f}" for sample in samples))


if __name__ == "__main__":
    main()
