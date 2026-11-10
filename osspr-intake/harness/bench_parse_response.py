"""Time TranslationParser.ProcessTranslation on one unlabelled batch.

A default-size batch whose model numbered the lines and separated them with a
blank line, and never wrote a Translation> tag. The unit tests already accept
that shape for two lines. The primary pattern cannot match it, but it
backtracks over the batch before the fallback does.
"""
import logging
import statistics
import sys
import time

from PySubtrans.Options import Options
from PySubtrans.Translation import Translation
from PySubtrans.TranslationParser import TranslationParser

logging.disable(logging.CRITICAL)

LINE = "She said they would wait until morning, because nobody was ready to leave."
BATCH = "\n\n".join(f"#{number}\n{LINE}" for number in range(1, 101))
RUNS = 7


def once() -> None:
    parser = TranslationParser("Translation", Options())
    parsed = parser.ProcessTranslation(Translation({"text": BATCH}))
    if parsed is None or len(parsed) != 100 or parsed[0].text != LINE or parsed[-1].number != 100:
        sys.exit("parsed batch did not match the unlabelled fallback")


def main() -> None:
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
