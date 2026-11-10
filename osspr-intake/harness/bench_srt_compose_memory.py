"""Peak memory of SrtFileHandler.compose for a dense two-hour SRT.

7200 cues, one a second. compose copies each cue into a new SubtitleLine and
an srt.Subtitle before joining the file. The input lines are built before
tracing, so the peak is the copies and the output string.
"""
import logging
import statistics
import sys
import tracemalloc
from datetime import timedelta

from PySubtrans.Formats.SrtFileHandler import SrtFileHandler
from PySubtrans.SubtitleData import SubtitleData
from PySubtrans.SubtitleLine import SubtitleLine

logging.disable(logging.CRITICAL)

CUE_COUNT = 7200
CUE_TEXT = "She said they would wait until morning, because nobody was ready."
RUNS = 7


def build_data() -> SubtitleData:
    lines = [
        SubtitleLine.Construct(index + 1, timedelta(seconds=index), timedelta(seconds=index + 1), CUE_TEXT)
        for index in range(CUE_COUNT)
    ]
    return SubtitleData(lines=lines, metadata={})


def main() -> None:
    data = build_data()
    handler = SrtFileHandler()
    handler.compose(data)

    peaks: list[int] = []
    for _ in range(RUNS):
        tracemalloc.start()
        tracemalloc.reset_peak()
        text = handler.compose(data)
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if text.count("-->") != CUE_COUNT or not text.startswith("1\n") or CUE_TEXT not in text:
            sys.exit("composed SRT did not contain the cues")
        peaks.append(peak)

    median = statistics.median(peaks)
    noise = (max(peaks) - min(peaks)) / median * 100.0
    print(f"baseline_value={int(median)}")
    print(f"noise_pct={noise:.4f}")
    print("runs_bytes=" + ",".join(str(peak) for peak in peaks))


if __name__ == "__main__":
    main()
