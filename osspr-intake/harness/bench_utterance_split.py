"""Wall time of UtteranceSplitter._best_split_index on unpunctuated chunks.

Each chunk is 180 words at 0.28s, about 50s, just under the default 60s local
chunk cap, with no sentence punctuation. _best_split_index is the scan
FitUtterance uses to place one cut. Twenty-four chunks are about twenty
minutes of that speech. The character limit is 42, so every chunk needs a split.

The score only needs the joined length of each side. The current scan rebuilds
both strings at every boundary.

Noise is (max - min) / median * 100 over five timed runs after one warmup.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from PySubtrans.Transcription.LineSettings import LineSettings
from PySubtrans.Transcription.UtteranceSplitter import UtteranceSplitter
from PySubtrans.Transcription.WordTiming import WordTiming

from common import fmt_ms, measure_ms

WORDS = (
    "the", "a", "of", "to", "and", "in", "is", "that", "it", "for", "was", "on",
    "are", "with", "as", "his", "they", "be", "at", "one", "have", "this", "from",
    "or", "had", "by", "not", "but", "what", "all", "were", "we", "when", "your",
    "can", "said", "there", "use", "an", "each", "which", "she", "do", "how",
    "their", "if", "will", "up", "other", "about", "out", "many", "then", "them",
    "these", "so", "some", "her", "would", "make", "like", "him", "into", "time",
    "has", "look", "two", "more", "write", "go", "see", "number", "no", "way",
    "could", "people", "my", "than", "first", "water", "been", "call", "who",
)

CHUNK_WORDS = 180
CHUNK_COUNT = 24
STEP_MS = 280


def build_chunks() -> list[list[WordTiming]]:
    chunks : list[list[WordTiming]] = []
    for chunk_index in range(CHUNK_COUNT):
        words : list[WordTiming] = []
        origin = chunk_index * CHUNK_WORDS * STEP_MS
        for index in range(CHUNK_WORDS):
            start = timedelta(milliseconds=origin + STEP_MS * index)
            words.append(WordTiming(
                text=WORDS[(chunk_index + index) % len(WORDS)],
                start=start,
                end=start + timedelta(milliseconds=280),
            ))
        chunks.append(words)
    return chunks


def main() -> None:
    chunks = build_chunks()
    splitter = UtteranceSplitter(LineSettings(
        max_line_chars=42,
        max_line_seconds=8.0,
        min_split_chars=4,
    ))

    def score() -> list[int]:
        indexes : list[int] = []
        for words in chunks:
            index = splitter._best_split_index(words)
            if index is None:
                raise RuntimeError("expected a split index")
            indexes.append(index)
        return indexes

    samples, median, noise = measure_ms(score)
    indexes = score()
    if len(indexes) != CHUNK_COUNT or any(index < 1 or index >= CHUNK_WORDS for index in indexes):
        raise SystemExit(f"unexpected split indexes {indexes}")

    digest = hashlib.sha256(",".join(str(index) for index in indexes).encode("utf-8")).hexdigest()[:16]
    print(fmt_ms(samples, median, noise))
    print(f"indexes={','.join(str(index) for index in indexes)}")
    print(f"checksum={digest}")


if __name__ == "__main__":
    main()
