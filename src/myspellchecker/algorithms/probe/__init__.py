"""Probe-based syllable-span detection for Myanmar spell checking.

Provides a frozen-encoder + thin-Linear-head detector that achieves +0.0067
composite when paired with rule-based correction strategies via the
ProbeBoostedCompoundStrategy and ProbeValidationStrategy.

See [[Probe Hybrid Ships at +0.0067 2026-05-03]] for design and benchmark
results.
"""

from myspellchecker.algorithms.probe.syllable_span_probe import (
    FrozenSyllableSpanProbe,
    ProbeInferenceEngine,
)

__all__ = ["FrozenSyllableSpanProbe", "ProbeInferenceEngine"]
