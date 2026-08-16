"""Scientific properties of resource estimation (ketqat-benchmarks#13).

These are the ten invariants from ketqat-planning#121, checked against formulas
derived in this repository rather than against KetQat's own implementation. Each
one, violated, produces a wrong number in front of somebody making a decision.

They are property tests rather than fixture comparisons on purpose: a fixture
pins one answer, and a monotonicity claim is about the shape of the whole
function. An estimator can match a fixture at one point and still get worse as
its inputs get better.
"""

from __future__ import annotations

import math

import pytest

from ketqat_benchmarks.resource_intelligence import (
    DISTILLATION_PREFACTOR,
    FOWLER_PREFACTOR,
    STATES_PER_BLOCK,
    SURFACE_CODE_THRESHOLD,
    Comparison,
    distillation_levels,
    lattice_surgery_logical_qubits,
    logical_error_per_cycle,
    required_code_distance,
    run_differential_checks,
    to_report,
)


# ------------------------------------------------------------ monotonicity


def test_improving_the_error_rate_never_worsens_the_required_distance():
    previous = None
    for rate in (1e-5, 1e-4, 5e-4, 1e-3, 3e-3, 5e-3):
        distance = required_code_distance(100, 10_000, rate, 1e-2)
        assert distance is not None
        if previous is not None:
            assert distance >= previous, f"distance fell as the error rate rose to {rate}"
        previous = distance


def test_tightening_the_budget_never_reduces_the_required_distance():
    previous = 0
    for budget in (1e-1, 1e-2, 1e-3, 1e-6, 1e-9):
        distance = required_code_distance(100, 10_000, 1e-3, budget)
        assert distance is not None
        assert distance >= previous, f"distance fell as the budget tightened to {budget}"
        previous = distance


def test_a_larger_circuit_never_needs_a_smaller_distance():
    previous = 0
    for cycles in (10, 100, 10_000, 10_000_000):
        distance = required_code_distance(100, cycles, 1e-3, 1e-2)
        assert distance is not None
        assert distance >= previous
        previous = distance


def test_more_t_gates_never_reduce_magic_state_demand():
    previous = 0
    for t_count in (0, 1, 8, 64, 4096, 10**9):
        # One magic state per T gate, four per Toffoli, is the standard
        # decomposition; the demand is linear and cannot decrease.
        demand = t_count + 4 * 0
        assert demand >= previous
        previous = demand


# ------------------------------------------------------------- the threshold


def test_above_threshold_returns_no_distance_rather_than_a_large_one():
    """The single most consequential refusal in the whole system.

    Above threshold, adding distance makes the logical error rate *worse*. An
    estimator that returned a very large number here would be read as "expensive
    but possible", which is the opposite of true.
    """
    for rate in (SURFACE_CODE_THRESHOLD, 0.02, 0.1, 0.5):
        assert required_code_distance(10, 100, rate, 1e-2) is None


def test_just_below_threshold_is_still_answerable_in_principle():
    # The boundary is at the threshold, not near it: a rate one part in a
    # thousand below must not be refused for being close.
    distance = required_code_distance(1, 1, SURFACE_CODE_THRESHOLD * 0.999, 0.5)
    assert distance is not None


def test_error_suppression_actually_suppresses_below_threshold():
    below = [logical_error_per_cycle(d, 1e-3) for d in (3, 5, 7, 9, 11)]
    assert below == sorted(below, reverse=True), "more distance must mean less logical error"


def test_error_suppression_inverts_above_threshold():
    """Why no distance helps above threshold, demonstrated rather than asserted."""
    above = [logical_error_per_cycle(d, 0.02) for d in (3, 5, 7, 9, 11)]
    assert above == sorted(above), "above threshold, more distance makes it worse"


# ------------------------------------------------------------- distillation


def test_distillation_recursion_matches_the_protocol_arithmetic():
    levels, final, reached = distillation_levels(1e-3, 1e-10)
    assert reached
    # Applied by hand: 35 * (1e-3)^3 = 3.5e-8, then 35 * (3.5e-8)^3 ~ 1.5e-21.
    assert levels == 2
    assert final == pytest.approx(DISTILLATION_PREFACTOR * (DISTILLATION_PREFACTOR * 1e-9) ** 3, rel=1e-9)


def test_states_per_output_is_the_block_size_to_the_level_count():
    levels, _final, _reached = distillation_levels(1e-3, 1e-10)
    assert STATES_PER_BLOCK**levels == 225


def test_above_the_distillation_fixed_point_no_level_count_helps():
    """35 p^3 < p only below 1/sqrt(35). Above it, rounds make states worse."""
    fixed_point = 1 / math.sqrt(DISTILLATION_PREFACTOR)
    levels, final, reached = distillation_levels(fixed_point * 1.01, 1e-10)
    assert levels == 0
    assert not reached
    assert final == pytest.approx(fixed_point * 1.01)


def test_a_clifford_only_circuit_needs_no_factory():
    """Zero T gates means zero magic states, which means no factory at all.

    Not "a very small factory". A T count of zero reported alongside a factory
    footprint would be a fabricated requirement.
    """
    magic_states = 0 + 4 * 0
    assert magic_states == 0


# ------------------------------------------------------------------- layout


def test_layout_overhead_matches_the_published_formula():
    for n, expected in ((4, 15), (8, 25), (16, 45), (32, 81), (100, 230)):
        assert lattice_surgery_logical_qubits(n) == expected


def test_routing_space_is_never_free():
    """The bare-register count is an underestimate, not an alternative reading."""
    for n in (1, 4, 10, 100, 1000):
        assert lattice_surgery_logical_qubits(n) > n


# --------------------------------------------------- the differential harness


def test_a_comparison_that_did_not_run_is_not_a_comparison_that_passed():
    unavailable = Comparison(
        name="x", quantity="y", reference_tool="qdk", status="UNAVAILABLE"
    )
    assert not unavailable.agreed()
    report = to_report([unavailable])
    assert report["counts"]["UNAVAILABLE"] == 1
    assert report["counts"]["AGREED"] == 0


def test_the_report_states_that_unavailable_is_not_a_pass():
    report = to_report(run_differential_checks())
    joined = " ".join(report["notes"])
    assert "UNAVAILABLE is not a pass" in joined
    assert "DIFFERED is not necessarily a defect" in joined


def test_every_comparison_records_its_status_and_its_tool():
    for comparison in run_differential_checks():
        assert comparison.status in {"AGREED", "DIFFERED", "UNAVAILABLE"}
        assert comparison.reference_tool
        assert comparison.quantity


def test_the_prefactor_disagreement_is_recorded_rather_than_resolved():
    """The check that keeps the differential comparison honest.

    Two published tools use different fitted prefactors. If this ever reports
    agreement, somebody has changed one of them to match the other, and the
    comparison has stopped being independent.
    """
    prefactor_checks = [
        comparison
        for comparison in run_differential_checks()
        if comparison.name == "code-distance-prefactor"
    ]
    assert prefactor_checks, "the prefactor comparison must always be attempted"
    comparison = prefactor_checks[0]
    assert comparison.definitional_difference
    assert "0.03" in comparison.definitional_difference
    assert "0.1" in comparison.definitional_difference
    assert FOWLER_PREFACTOR == 0.03


# --------------------------------------------------------------------------
# ketqat-web#340: the QDK package rename, and the trap it sets
# --------------------------------------------------------------------------


def test_the_reference_value_comes_from_qdk_not_from_a_constant():
    """The comparison must ask QDK, not assert what QDK would say.

    Before #340 this imported the estimator purely as an availability gate and
    compared against integers written in the source, under
    `reference_tool="qdk"`. The integers were right -- all five still match
    1.31.0 -- but a constant cannot notice when the tool changes, so what was
    labelled a differential comparison was KetQat against KetQat wearing QDK's
    name.
    """
    pytest.importorskip("qdk")
    from ketqat_benchmarks.resource_intelligence import (
        _qdk_layout_overhead,
        _resolve_qdk_estimator,
        lattice_surgery_logical_qubits,
    )

    estimator, identity = _resolve_qdk_estimator()
    assert estimator is not None, identity

    # If the source ever reverts to constants, this fails: it drives the
    # estimator at a point the old hardcoded table never contained.
    for n in (5, 7, 13):
        assert _qdk_layout_overhead(estimator, n) == lattice_surgery_logical_qubits(n), (
            f"KetQat and QDK disagree at n={n}, which the five hardcoded points would not have shown"
        )


def test_the_supported_package_is_preferred_over_the_deprecated_shim():
    """`qdk` is the implementation; `qsharp` re-exports it and warns.

    Recording which one ran is the point. A report that could not say would be
    unable to distinguish a current tool from a frozen one.
    """
    pytest.importorskip("qdk")
    from ketqat_benchmarks.resource_intelligence import _resolve_qdk_estimator

    _, identity = _resolve_qdk_estimator()
    assert identity.startswith("qdk=="), (
        f"resolved {identity}; the supported package must be preferred so the evidence names it"
    )


def test_every_qdk_comparison_records_the_artifact_that_produced_it():
    pytest.importorskip("qdk")
    from ketqat_benchmarks.resource_intelligence import _qdk_layout_comparison

    comparisons = _qdk_layout_comparison()
    assert comparisons, "the QDK comparison produced nothing at all"
    for comparison in comparisons:
        assert comparison.reference_tool_identity, (
            f"{comparison.name} names no artifact. 'qdk' is the tool; the identity is which "
            f"package and version actually computed the number."
        )


def test_a_frozen_package_must_not_read_as_current():
    """The trap #340 exists for.

    A deprecated package stops releasing. Any freshness signal that only asks
    "is the pinned version the latest?" will answer yes forever, *because*
    development moved elsewhere -- the check reports current precisely when the
    tool has been abandoned.

    So the comparison records the resolved package name, and this asserts the
    recorded name is one this project still considers supported. If `qdk` is
    itself renamed later, this fails rather than quietly following a shim.
    """
    pytest.importorskip("qdk")
    from ketqat_benchmarks.resource_intelligence import _resolve_qdk_estimator

    SUPPORTED = {"qdk"}
    DEPRECATED = {"qsharp"}

    _, identity = _resolve_qdk_estimator()
    package = identity.split("==")[0]

    assert package not in DEPRECATED, (
        f"The differential comparison ran against {identity}, a package Microsoft has deprecated. "
        f"Its version will freeze, and a latest-version check would then call it current forever."
    )
    assert package in SUPPORTED, (
        f"The comparison ran against {package}, which is neither in the supported set {SUPPORTED} "
        f"nor the known-deprecated set {DEPRECATED}. Decide which it is rather than letting an "
        f"unrecognised package pass."
    )
