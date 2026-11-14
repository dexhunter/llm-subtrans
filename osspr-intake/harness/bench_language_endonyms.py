"""Peak memory of the first build of the native-language name map.

Resolving a name such as 中文 or Deutsch misses the English map and builds
_language_endonyms. That loads every Babel locale and keeps the parsed CLDR
trees in Babel's cache. The measured peak is those trees, not the 308-name map.
"""
import gc
import statistics
import sys
import tracemalloc

import babel.localedata as localedata

from PySubtrans.Helpers import Languages

RUNS = 7
EXPECTED = {
    "deutsch": "de",
    "español": "es",
    "中文": "zh",
    "日本語": "ja",
    "한국어": "ko",
}


def reset_locale_cache() -> None:
    Languages._endonyms = None
    localedata._cache.clear()


def main() -> None:
    # One unmeasured call loads Babel's global tables, which stay cached.
    # Each measured call reloads the locale trees the function retains.
    Languages._language_endonyms()
    reset_locale_cache()

    peaks : list[int] = []
    for _run in range(RUNS):
        gc.collect()
        tracemalloc.start()
        tracemalloc.reset_peak()
        names = Languages._language_endonyms()
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        for name, code in EXPECTED.items():
            if names.get(name.casefold()) != code:
                sys.exit(f"endonym map mismatch for {name!r}")
        peaks.append(peak)
        reset_locale_cache()

    median = statistics.median(peaks)
    noise = (max(peaks) - min(peaks)) / median * 100.0
    print(f"baseline_value={int(median)}")
    print(f"noise_pct={noise:.4f}")
    print("runs_bytes=" + ",".join(str(peak) for peak in peaks))


if __name__ == "__main__":
    main()
