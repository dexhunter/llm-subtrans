"""Time AudioExtractor.QuietestStretch across a long clip with no pauses.

The default quiet scan looks 5s back from a 60s chunk cap, and only when no
silence was found. A two-hour clip is 120 of those scans. Each one sums a
fresh slice of every 10ms frame in Python.
"""
import array
import io
import logging
import math
import statistics
import sys
import time
import wave

from PySubtrans.SettingsType import SettingsType
from PySubtrans.Transcription.AudioExtractor import AudioExtractor

logging.disable(logging.CRITICAL)

SCAN_SECONDS = 5.0
STRETCH_SECONDS = 0.3
SCANS = 120
RUNS = 7


def tone_wav(seconds: float, sample_rate: int = 16000) -> bytes:
    """5s of 16kHz mono PCM with a quieter dip near the end, so the stretch is real."""
    count = int(sample_rate * seconds)
    samples = array.array("h", (int(8000 * math.sin(index * 0.15)) for index in range(count)))
    dip = int(sample_rate * (seconds - 0.8))
    for index in range(dip, dip + int(sample_rate * 0.4)):
        samples[index] = samples[index] // 8

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())
    return buffer.getvalue()


def main() -> None:
    audio = tone_wav(SCAN_SECONDS)
    extractor = AudioExtractor(SettingsType())
    expected = extractor.QuietestStretch(audio, STRETCH_SECONDS)
    if expected is None:
        sys.exit("quiet scan returned no stretch")

    def once() -> None:
        found = None
        for _ in range(SCANS):
            found = extractor.QuietestStretch(audio, STRETCH_SECONDS)
        if found != expected:
            sys.exit("quiet scan was not stable")

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
