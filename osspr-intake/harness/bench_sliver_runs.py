"""Time LineMerger._sliver_runs on one word-level chunk.

2400 words, one every 0.26s and each 0.24s long, is a 10.4 minute same-speaker
chunk. Every line is shorter than the 0.8s minimum, and the 0.02s gap is inside
the same-speaker merge window, so they stay one run. Each new line currently
rescans that run for its latest end.
"""
import logging
import statistics
import sys
import time
from datetime import timedelta

from PySubtrans.Transcription.LineMerger import LineMerger
from PySubtrans.Transcription.LineSettings import LineSettings
from PySubtrans.Transcription.TranscriptionSegment import TranscriptionSegment

logging.disable(logging.CRITICAL)

WORD_COUNT = 2400
RUNS = 7


def build_lines() -> list[TranscriptionSegment]:
    lines: list[TranscriptionSegment] = []
    cursor = 0.0
    for index in range(WORD_COUNT):
        lines.append(TranscriptionSegment(
            start=timedelta(seconds=cursor),
            end=timedelta(seconds=cursor + 0.24),
            text=f"word{index % 50}",
            speaker="Ann",
        ))
        cursor += 0.26
    return lines


def main() -> None:
    lines = build_lines()
    merger = LineMerger(LineSettings(max_line_chars=120, max_line_seconds=4.0, min_line_seconds=0.8))

    def once() -> None:
        runs = merger._sliver_runs(lines)
        if len(runs) != 1 or len(runs[0]) != WORD_COUNT:
            sys.exit("word-level chunk did not stay one sliver run")

    once()
    samples: list[float] = []
    for _ in range(RUNS):
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
