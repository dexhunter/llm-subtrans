"""cl100k token count of one TranslationPrompt.GenerateBatchPrompt batch.

A default batch is 100 lines (max_batch_size). Each line is a short spoken
sentence with an alternating speaker. Context is a description, names,
terminology, and ten history lines, which is max_context_summaries.

The count uses the vendored cl100k_base BPE file (GPT-4 / gpt-3.5-turbo),
loaded from disk so the run does not touch the network. The repo does not
ship a tokenizer of its own. Meaning of the prompt is the lines, speakers,
and context values; the repeated Original>/Translation> scaffolding is the
part a local change could shrink without changing those values.

Noise is (max - min) / median * 100 over five counts after one warmup.
"""

from __future__ import annotations

import hashlib
import os
import statistics
from datetime import timedelta

import tiktoken
from tiktoken.load import load_tiktoken_bpe

from PySubtrans.SubtitleLine import SubtitleLine
from PySubtrans.TranslationPrompt import TranslationPrompt

from common import spread_pct

# Pattern and special tokens copied from tiktoken's cl100k_base constructor.
CL100K_PAT = (
    r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
)
CL100K_HASH = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
LINE_COUNT = 100

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


def encoder() -> tiktoken.Encoding:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cl100k_base.tiktoken")
    ranks = load_tiktoken_bpe(path, expected_hash=CL100K_HASH)
    return tiktoken.Encoding(
        name="cl100k_base",
        pat_str=CL100K_PAT,
        mergeable_ranks=ranks,
        special_tokens={
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    )


def line_text(index : int) -> str:
    tokens = [WORDS[(index + offset) % len(WORDS)] for offset in range(12)]
    tokens[0] = tokens[0].capitalize()
    return " ".join(tokens) + "."


def build_lines() -> list[SubtitleLine]:
    lines : list[SubtitleLine] = []
    for index in range(LINE_COUNT):
        start = timedelta(seconds=index * 2)
        lines.append(SubtitleLine.Construct(
            index + 1,
            start,
            start + timedelta(milliseconds=1800),
            line_text(index),
            {"speaker": "Amina" if index % 2 == 0 else "Jonah"},
        ))
    return lines


def build_context() -> dict:
    return {
        "description": "Two colleagues argue in a kitchen after the guests have left.",
        "names": ["Amina", "Jonah", "Mrs. Adler"],
        "terminology": ["the Agency", "platform nine"],
        "history": [
            f"scene {scene}: they argued about the missing keys and the last train"
            for scene in range(1, 11)
        ],
        "scene": "Scene 4: the kitchen, late",
        "summary": "They are deciding whether to leave tonight.",
        "batch": "Batch 3 of scene 4",
    }


def main() -> None:
    enc = encoder()
    lines = build_lines()
    context = build_context()
    prompt = TranslationPrompt("Translate these subtitles into Spanish.", True)

    def count() -> int:
        text = prompt.GenerateBatchPrompt(lines, context)
        return len(enc.encode(text))

    # Warmup, then five counts. The count is deterministic; the repeats prove it.
    count()
    samples = [count() for _repeat in range(5)]
    text = prompt.GenerateBatchPrompt(lines, context)
    body = "\n".join(line.text or "" for line in lines)
    if samples[0] != len(enc.encode(text)):
        raise SystemExit("token count changed between runs")
    for line in lines:
        if line.text not in text:
            raise SystemExit("prompt dropped a source line")

    median = int(statistics.median(samples))
    noise = spread_pct([float(sample) for sample in samples])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    print(f"samples={samples}")
    print(f"baseline_value={median}")
    print(f"noise_pct={noise:.4f}")
    print(f"body_tokens={len(enc.encode(body))}")
    print(f"prompt_chars={len(text)}")
    print(f"checksum={digest}")


if __name__ == "__main__":
    main()
