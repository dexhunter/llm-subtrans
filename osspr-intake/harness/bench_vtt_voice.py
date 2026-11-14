"""Time VttFileHandler._process_vtt_text across one voiced two-hour file.

7200 cues, one a second, each wrapped in a single voice tag the way a
YouTube or recorded WebVTT names the speaker. parse_string calls this once
per cue. The cue strings are built before the timer.
"""
import gc
import statistics
import sys
import time

from PySubtrans.Formats.VttFileHandler import VttFileHandler

CUE_COUNT = 7200
RUNS = 7
PARTIAL = "<v Esme>Hee!</v> <i>laughter</i>"


def build_cues() -> list[str]:
    return [
        f"<v.loud Narrator>This is cue number {index} about the weather.</v>"
        for index in range(CUE_COUNT)
    ]


def main() -> None:
    cues = build_cues()
    handler = VttFileHandler()

    def once() -> list[tuple[str, dict]]:
        return [handler._process_vtt_text(cue) for cue in cues]

    processed = once()
    first_text, first_meta = processed[0]
    last_text, last_meta = processed[-1]
    partial_text, partial_meta = handler._process_vtt_text(PARTIAL)
    if first_text != "This is cue number 0 about the weather." or first_meta.get("speaker") != "Narrator":
        sys.exit("first voice cue was not stripped")
    if first_meta.get("voice_classes") != ["loud"]:
        sys.exit("first voice cue lost its class")
    if last_text != f"This is cue number {CUE_COUNT - 1} about the weather." or last_meta.get("speaker") != "Narrator":
        sys.exit("last voice cue was not stripped")
    if partial_text != PARTIAL or partial_meta:
        sys.exit("a voice tag that does not wrap the cue was rewritten")

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
