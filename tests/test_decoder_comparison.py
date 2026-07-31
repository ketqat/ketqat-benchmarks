"""The comparison must compare decoders, not samples (ketqat-benchmarks#1)."""
from __future__ import annotations

import pytest

from ketqat_benchmarks.decoder_comparison import DecoderOutcome, run_comparison


def test_rate_is_none_when_nothing_was_decoded() -> None:
    # 0.0 from zero shots is indistinguishable in a table from a perfect decoder.
    assert DecoderOutcome("x", "y", "1", False).logical_error_rate is None
    assert DecoderOutcome("x", "y", "1", False).wilson_interval() is None


def test_abstentions_are_excluded_from_the_denominator() -> None:
    # Scoring an abstention as a failure flatters whichever decoder never abstains.
    outcome = DecoderOutcome("x", "y", "1", True, shots=100, logical_errors=5, abstentions=50)
    assert outcome.logical_error_rate == pytest.approx(0.10)


def test_wilson_stays_valid_at_zero_failures() -> None:
    # The normal approximation gives [0, 0] here, asserting certainty from the one
    # observation that provides none.
    outcome = DecoderOutcome("x", "y", "1", True, shots=10_000)
    low, high = outcome.wilson_interval()
    assert low == 0.0
    assert 0.0 < high < 0.001


def test_every_decoder_sees_the_same_samples() -> None:
    pytest.importorskip("stim")
    report = run_comparison(distance=3, rounds=3, noise=0.02, shots=400, seed=11)
    ran = [d for d in report["decoders"] if d["available"]]
    if len(ran) < 2:
        pytest.skip("needs at least two decoders installed")
    # Identical shot counts is the observable consequence of one shared sample set.
    assert len({d["shots"] for d in ran}) == 1
    assert report["is_demo"] is False
    assert report["is_leaderboard"] is (len(ran) >= 2)
    assert report["comparability"]["sample_source"] == "single shared Stim detector sample"


def test_a_missing_decoder_is_not_run_rather_than_zero() -> None:
    outcome = DecoderOutcome("absent", "nothing", "", False, not_run_reason="No module")
    assert outcome.logical_error_rate is None
    assert outcome.not_run_reason
