"""Time LineMerger._balanced_chunks on one same-speaker word run.

2400 one-word lines, each 0.24s long and 0.26s after the previous end.
That is one 20-minute word-level chunk whose gaps sit inside the same-speaker
merge window. Defaults are 120 characters and 4.0 seconds, so the run has to
be divided. The lines are built before the timer.
"""
import gc
import statistics
import sys
import time
from datetime import timedelta

from PySubtrans.Transcription.LineMerger import LineMerger
from PySubtrans.Transcription.LineSettings import LineSettings
from PySubtrans.Transcription.TranscriptionSegment import TranscriptionSegment

WORDS = (
    "she", "waited", "by", "the", "door", "until", "the", "rain",
    "stopped", "and", "then", "walked", "home", "alone", "again",
)
RUN_LENGTH = 2400
RUNS = 7


def build_run() -> list[TranscriptionSegment]:
    lines : list[TranscriptionSegment] = []
    for index in range(RUN_LENGTH):
        start = index * 0.5
        lines.append(TranscriptionSegment(
            start=timedelta(seconds=start),
            end=timedelta(seconds=start + 0.24),
            text=WORDS[index % len(WORDS)],
            speaker="0",
        ))
    return lines


def main() -> None:
    run = build_run()
    merger = LineMerger(LineSettings(max_line_chars=120, max_line_seconds=4.0))

    def once() -> list[list[TranscriptionSegment]]:
        return merger._balanced_chunks(run)

    chunks = once()
    if len(chunks) != 300 or any(len(chunk) != 8 for chunk in chunks):
        sys.exit(f"expected 300 chunks of 8 lines, got {len(chunks)}")
    if chunks[0][0].text != WORDS[0] or chunks[-1][-1].text != WORDS[(RUN_LENGTH - 1) % len(WORDS)]:
        sys.exit("chunk ends were not the run's first and last words")
    if not all(merger._chunk_fits(chunk) for chunk in chunks):
        sys.exit("a returned chunk does not fit the line limits")

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
