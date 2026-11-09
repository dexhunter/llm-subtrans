"""Wall time of WordAlignment.AlignWords on one long verbatim chunk.

Gemini and ElevenLabs send up to 600-900s of audio as a single chunk.
This is about ten minutes of English (2400 words at 0.26s each). The transcript
is those words with a full stop every 18 words, which is how a timed word list
lines up with a punctuated transcript.

Noise is (max - min) / median * 100 over five timed runs after one warmup.
"""

from __future__ import annotations

import hashlib
from datetime import timedelta

from PySubtrans.Transcription.WordAlignment import AlignWords
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
    "oil", "its", "now", "find", "long", "down", "day", "did", "get", "come",
    "made", "may", "part", "over", "new", "sound", "take", "only", "little",
    "work", "know", "place", "year", "live", "me", "back", "give", "most", "very",
    "after", "thing", "our", "just", "name", "good", "sentence", "man", "think",
    "say", "great", "where", "help", "through", "much", "before", "line", "right",
    "too", "mean", "old", "any", "same", "tell", "boy", "follow", "came", "want",
    "show", "also", "around", "form", "three", "small", "set", "put", "end",
    "does", "another", "well", "large", "must", "big", "even", "such", "because",
    "turn", "here", "why", "ask", "went", "men", "read", "need", "land",
    "different", "home", "us", "move", "try", "kind", "hand", "picture", "again",
    "change", "off", "play", "spell", "air", "away", "animal", "house", "point",
    "page", "letter", "mother", "answer", "found", "study", "still", "learn",
    "should", "world",
)

WORD_COUNT = 2400
STEP_MS = 260


def build_chunk() -> tuple[str, list[WordTiming]]:
    """A 10.4 minute chunk: spoken words, plus sentence stops in the transcript only."""
    parts : list[str] = []
    words : list[WordTiming] = []
    for index in range(WORD_COUNT):
        token = WORDS[index % len(WORDS)]
        parts.append(token + ("." if index and index % 18 == 0 else ""))
        start = timedelta(milliseconds=STEP_MS * index)
        words.append(WordTiming(text=token, start=start, end=start + timedelta(milliseconds=240)))
    return " ".join(parts), words


def digest(aligned : list) -> str:
    payload = "\n".join(
        f"{item.start}|{item.end}|{item.matched}|{item.word.text}|{item.word.start.total_seconds():.3f}"
        for item in aligned
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    text, words = build_chunk()

    def align() -> list:
        return AlignWords(text, words)

    samples, median, noise = measure_ms(align)
    aligned = align()
    if len(aligned) != WORD_COUNT:
        raise SystemExit(f"expected {WORD_COUNT} aligned words, got {len(aligned)}")

    print(fmt_ms(samples, median, noise))
    print(f"aligned={len(aligned)}")
    print(f"transcript_chars={len(text)}")
    print(f"checksum={digest(aligned)}")


if __name__ == "__main__":
    main()
