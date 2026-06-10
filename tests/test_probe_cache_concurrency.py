"""Concurrency regression test for ProbeInferenceEngine.score_sentence.

The single probe engine is shared across all probe strategies and is hit by
check_batch_async via asyncio.to_thread workers. Before the lat-02 cache was
guarded, an unsynchronized OrderedDict LRU (get/move_to_end racing another
thread's popitem(last=False)) could raise KeyError near the size cap. This test
drives many threads against a near-full cache with distinct per-thread texts to
force concurrent evictions, and asserts no exception plus correct results.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from myspellchecker.algorithms.probe.syllable_span_probe import (
    ProbeInferenceEngine,
    _SyllableSpan,
)


def _make_engine(cache_max: int) -> ProbeInferenceEngine:
    """Build an engine without loading the model (bypass __init__).

    Only the cache machinery is needed; _score_sentence_uncached is stubbed to
    a deterministic pure function so the test never touches torch / the encoder.
    """
    eng = ProbeInferenceEngine.__new__(ProbeInferenceEngine)
    eng._score_cache = OrderedDict()
    eng._SCORE_CACHE_MAX = cache_max
    eng._score_cache_lock = threading.Lock()

    def _uncached(text: str) -> tuple[list[float], list[_SyllableSpan]]:
        # Deterministic per-text payload: a probe is frozen/deterministic, so a
        # given text must always map to the same scores regardless of thread.
        probs = [float(len(text) % 7) / 7.0]
        spans = [_SyllableSpan(text=text, start=0, end=len(text))]
        return probs, spans

    eng._score_sentence_uncached = _uncached  # type: ignore[method-assign]
    return eng


def test_concurrent_eviction_no_keyerror() -> None:
    eng = _make_engine(cache_max=256)
    # Pre-fill to the cap so every miss triggers a popitem under contention.
    for i in range(256):
        eng.score_sentence(f"seed-{i}")
    assert len(eng._score_cache) == 256

    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def worker(tid: int) -> None:
        barrier.wait()  # maximize overlap on the LRU ops
        try:
            for j in range(400):
                text = f"t{tid}-{j}"
                probs, spans = eng.score_sentence(text)
                assert spans and spans[0].text == text
                assert probs == [float(len(text) % 7) / 7.0]
                # Re-read a hot key to exercise the get/move_to_end path.
                eng.score_sentence("seed-0")
        except BaseException as exc:  # noqa: BLE001 - capture for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"score_sentence raced under concurrency: {errors[:3]}"
    # Invariant: the LRU never exceeds its cap despite concurrent inserts.
    assert len(eng._score_cache) <= 256


def test_cache_hit_returns_independent_copies() -> None:
    eng = _make_engine(cache_max=8)
    p1, s1 = eng.score_sentence("ကျောင်း")
    p2, s2 = eng.score_sentence("ကျောင်း")
    # Outer lists are fresh per call so a caller mutating one cannot poison the
    # cache for the next caller.
    assert p1 == p2 and s1 == s2
    assert p1 is not p2
    assert s1 is not s2
    p1.append(999.0)
    p3, _ = eng.score_sentence("ကျောင်း")
    assert 999.0 not in p3
