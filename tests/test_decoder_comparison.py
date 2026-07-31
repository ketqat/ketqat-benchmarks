"""The comparison must compare decoders, not samples — and prove it."""
from __future__ import annotations

import math

import pytest

from ketqat_benchmarks.decoder_comparison import (
    mcnemar_exact_p,
    paired_risk_difference,
    run_comparison,
    wilson,
)


def test_wilson_stays_valid_at_zero_failures() -> None:
    low, high = wilson(0, 10_000)
    assert low == 0.0
    assert 0.0 < high < 0.001


def test_mcnemar_symmetric_and_null_at_equal_discordants() -> None:
    assert mcnemar_exact_p(5, 5) == pytest.approx(mcnemar_exact_p(5, 5))
    assert mcnemar_exact_p(0, 0) == 1.0
    # Strong asymmetry must be significant.
    assert mcnemar_exact_p(50, 5) < 1e-6
    assert mcnemar_exact_p(50, 5) == mcnemar_exact_p(5, 50)


def test_mcnemar_survives_large_discordant_counts() -> None:
    # 2.0**n overflows a float past n ~ 1024; the first version died here on real data.
    p = mcnemar_exact_p(679, 446)
    assert 0.0 <= p < 1e-8
    p2 = mcnemar_exact_p(600, 600)
    assert p2 > 0.9


def test_exact_and_approximate_mcnemar_agree_at_the_boundary() -> None:
    # The exact path runs to n=1000, the approximation beyond. They must agree where
    # they meet, or the boundary itself changes conclusions.
    exact = mcnemar_exact_p(530, 470)   # n = 1000, exact
    approx = mcnemar_exact_p(531, 470)  # n = 1001, approximated
    assert abs(exact - approx) < 0.02


def test_paired_ci_driven_by_discordant_counts() -> None:
    # 100 shots, both wrong on the same 10: no discordance, difference exactly zero
    # with zero-width CI. Concordant errors tell us nothing about the difference.
    both = [True] * 10 + [False] * 90
    stats = paired_risk_difference(both, both)
    assert stats["risk_difference"] == 0.0
    assert stats["b01"] == 0 and stats["b10"] == 0
    assert stats["ci_95"][0] == pytest.approx(0.0)
    assert stats["ci_95"][1] == pytest.approx(0.0)


def test_abstention_cannot_improve_rank() -> None:
    stim = pytest.importorskip("stim")  # noqa: F841
    report = run_comparison(3, 3, 0.02, 400, seed=11, with_timing=False)
    ran = [d for d in report["decoders"] if d["available"]]
    if len(ran) < 2:
        pytest.skip("needs two decoders")
    for row in ran:
        # unconditional risk counts abstentions as failures for ranking purposes
        expected = (row["logical_errors"] + row["abstentions"]) / row["shots"]
        assert row["unconditional_risk"] == pytest.approx(expected)
    ranks = sorted(ran, key=lambda r: r["rank_by_unconditional_risk"])
    risks = [r["unconditional_risk"] for r in ranks]
    assert risks == sorted(risks)


def test_every_decoder_provably_consumed_the_shared_sample() -> None:
    pytest.importorskip("stim")
    report = run_comparison(3, 3, 0.02, 300, seed=5, with_timing=False)
    master = report["experiment"]["sample_sha256"]
    ran = [d for d in report["decoders"] if d["available"]]
    assert len(ran) >= 2
    for row in ran:
        # The digest each adapter computed from the arrays it iterated, not a copy of
        # the master hash: equality is the proof of consumption.
        assert row["consumed_sample_sha256"] == master
    assert report["experiment"]["sampling"] == "direct shared-Stim detector sampling (not Sinter)"


def test_is_demo_and_publishable_are_derived() -> None:
    pytest.importorskip("stim")
    report = run_comparison(3, 3, 0.02, 200, seed=3, with_timing=False)
    ran = [d for d in report["decoders"] if d["available"]]
    assert report["is_demo"] is (len(ran) == 0)
    if len(ran) >= 2:
        assert report["publishable"] is True
    assert "reproducibility_sha256" in report
    assert report["execution_class"] == "SIMULATION"


def test_paired_comparisons_carry_multiplicity_correction() -> None:
    pytest.importorskip("stim")
    report = run_comparison(3, 3, 0.02, 300, seed=2, with_timing=False)
    for pair in report["paired_comparisons"]:
        assert pair["alpha_bonferroni"] <= 0.05 / max(1, len(report["paired_comparisons"]))
        assert "mcnemar_p" in pair and "ci_95" in pair
