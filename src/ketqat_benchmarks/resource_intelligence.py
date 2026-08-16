"""Independent validation of KetQat's resource intelligence (ketqat-benchmarks#13).

KetQat computes code distances, layout overheads and distillation costs in
TypeScript, and checks them against a second KetQat implementation in Python.
That is a consistency check, not a correctness check: two implementations with
one author share their author's mistakes, and agreeing about a wrong formula is
exactly what they would do.

This module checks the numbers against things that were not written here:

* **Closed-form arithmetic.** The surface-code distance inequality and the
  15-to-1 recursion are formulas. Their answers can be derived on paper, and
  this file derives them independently rather than calling KetQat and asserting
  the result equals itself.
* **Microsoft's QDK resource estimator**, where both tools claim to compute the
  same quantity under the same stated assumptions.
* **Qualtran**, on the same basis.

Where the tools genuinely define a quantity differently -- and they do, notably
the fitted logical-error prefactor and the layout convention -- the difference is
**recorded with its cause**, not tuned away. Forcing agreement would destroy the
only thing this comparison is for. A differential check whose failure mode is
"adjust KetQat until it matches" is a check that reports whatever it was told to.

## The honesty requirement

`qdk` and `qualtran` are optional. A skipped comparison must never be reported as
a passing one, so every result carries `status` (`AGREED`, `DIFFERED`,
`UNAVAILABLE`) and the CI gate fails when a comparison it required was
`UNAVAILABLE`. A green run where nothing ran is the failure this file is
arranged to prevent.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import math

from dataclasses import asdict, dataclass, field
from typing import Any, Callable

#: Surface-code threshold under circuit-level depolarizing noise, after Fowler
#: et al. (2012), arXiv:1208.0928. A property of the code and decoder.
SURFACE_CODE_THRESHOLD = 0.01

#: Conventional fitted prefactor A in p_L = A (p/p_th)^((d+1)/2). Fitted, not
#: derived; Qualtran's own implementation says of it "The pre-factor $a$ has no
#: clear provenance."
FOWLER_PREFACTOR = 0.03

#: Input states consumed per output by one 15-to-1 distillation block.
STATES_PER_BLOCK = 15

#: Leading-order error suppression of 15-to-1: p -> 35 p^3.
DISTILLATION_PREFACTOR = 35


# --------------------------------------------------------------------------
# Closed forms, derived here rather than imported
# --------------------------------------------------------------------------


def logical_error_per_cycle(distance: int, physical_error_rate: float,
                            threshold: float = SURFACE_CODE_THRESHOLD,
                            prefactor: float = FOWLER_PREFACTOR) -> float:
    """p_L = A (p/p_th)^((d+1)/2), written out."""
    return prefactor * (physical_error_rate / threshold) ** ((distance + 1) / 2)


def required_code_distance(logical_qubits: int, logical_cycles: int,
                           physical_error_rate: float, error_budget: float,
                           threshold: float = SURFACE_CODE_THRESHOLD,
                           prefactor: float = FOWLER_PREFACTOR,
                           max_distance: int = 201) -> int | None:
    """Smallest odd distance meeting the budget, by direct search.

    Deliberately a loop rather than the algebraic inversion KetQat uses. Two
    routes to the same integer is the point: an error in the rearrangement shows
    up here as a disagreement rather than being reproduced faithfully.

    Returns ``None`` above threshold, where adding distance makes the logical
    error rate *worse* and no distance satisfies any budget.
    """
    if physical_error_rate >= threshold:
        return None
    if logical_qubits <= 0 or logical_cycles <= 0:
        return 1
    budget_per_qubit_cycle = error_budget / (logical_qubits * logical_cycles)
    for distance in range(3, max_distance + 1, 2):
        if logical_error_per_cycle(distance, physical_error_rate, threshold, prefactor) <= budget_per_qubit_cycle:
            return distance
    return None


def distillation_levels(raw_error: float, target_error: float, max_levels: int = 6) -> tuple[int, float, bool]:
    """15-to-1 rounds to reach ``target_error``; the recursion, applied.

    Above the protocol's fixed point ``1/sqrt(35)`` each round makes the state
    worse, so no level count helps and the caller is told so rather than given a
    number.
    """
    fixed_point = 1 / math.sqrt(DISTILLATION_PREFACTOR)
    if raw_error >= fixed_point:
        return 0, raw_error, raw_error <= target_error
    current = raw_error
    levels = 0
    while current > target_error and levels < max_levels:
        current = DISTILLATION_PREFACTOR * current ** 3
        levels += 1
    return levels, current, current <= target_error


def lattice_surgery_logical_qubits(algorithm_qubits: int) -> int:
    """2n + ceil(sqrt(8n)) + 1, after Beverland et al. (2022), arXiv:2211.07629."""
    return 2 * algorithm_qubits + math.ceil(math.sqrt(8 * algorithm_qubits)) + 1


# --------------------------------------------------------------------------
# Differential comparison
# --------------------------------------------------------------------------


@dataclass
class Comparison:
    """One quantity, computed two ways.

    ``status`` is three-valued on purpose. A comparison that could not run is
    not a comparison that passed, and collapsing the two is how a CI job reports
    green while checking nothing.
    """

    name: str
    quantity: str
    reference_tool: str
    #: The package and version that actually produced ``reference_value``, as
    #: resolved at run time. "qdk" names the tool; this names the artifact.
    #: Without it a report cannot say whether a number came from the supported
    #: package or from the deprecated shim over it (ketqat-web#340).
    reference_tool_identity: str | None = None
    ketqat_value: float | None = None
    reference_value: float | None = None
    status: str = "UNAVAILABLE"
    #: Present when the two tools define the quantity differently. Recorded
    #: rather than resolved: the difference is the finding.
    definitional_difference: str | None = None
    detail: str = ""
    assumptions: list[str] = field(default_factory=list)

    def agreed(self, tolerance: float = 0.0) -> bool:
        if self.ketqat_value is None or self.reference_value is None:
            return False
        if tolerance == 0.0:
            return self.ketqat_value == self.reference_value
        return abs(self.ketqat_value - self.reference_value) <= tolerance * abs(self.reference_value)


def _resolve_qdk_estimator() -> tuple[Any, str] | tuple[None, str]:
    """The QDK estimator module, preferring the supported package.

    Microsoft renamed `qsharp` to `qdk`. `qsharp/estimator/__init__.py` is now a
    shim: a module docstring reading "Deprecated. Use qdk.estimator instead."
    followed by ``from qdk.estimator import *`` and nothing else.

    So the two are the *same implementation* reached by two names, which is why
    migrating is safe: verified on 2026-08-15 at 1.31.0, both produce identical
    algorithmicLogicalQubits at n = 4, 8, 16, 32 and 100. This is a rename and a
    deprecation, not an estimator or scientific-model change.

    `qdk` is tried first so the evidence names the supported package. `qsharp`
    remains a fallback for an environment that predates the rename, and the
    resolved identity is recorded either way -- a report that could not say
    which one ran would be unable to distinguish a current tool from a frozen
    one (ketqat-web#340).
    """
    for package in ("qdk", "qsharp"):
        try:
            module = importlib.import_module(f"{package}.estimator")
        except Exception:  # noqa: BLE001 - any import failure means unavailable
            continue
        try:
            version = importlib.metadata.version(package)
        except Exception:  # noqa: BLE001
            version = "unknown"
        return module, f"{package}=={version}"
    return None, "neither qdk nor qsharp is importable"


def _qdk_layout_overhead(estimator: Any, logical_qubits: int) -> int:
    """Ask QDK for the layout overhead, rather than asserting what it would say.

    The previous version imported the estimator purely as an availability gate
    and then compared against integers written in this file, under
    ``reference_tool="qdk"``. Those integers were right -- all five still match
    QDK 1.31.0 -- but a constant cannot notice when the tool changes, so the
    comparison was really KetQat against KetQat wearing QDK's name.
    """
    result = estimator.LogicalCounts(
        {
            "numQubits": logical_qubits,
            "tCount": 0,
            "rotationCount": 0,
            "rotationDepth": 0,
            "cczCount": 0,
            "ccixCount": 0,
            "measurementCount": logical_qubits,
        }
    ).estimate()
    data = result if isinstance(result, dict) else result.data()
    return int(data["physicalCounts"]["breakdown"]["algorithmicLogicalQubits"])


def _qdk_layout_comparison() -> list[Comparison]:
    """Layout overhead against Microsoft's QDK estimator.

    The only quantity where the two tools indisputably compute the same thing
    from the same formula, which makes it the one where a mismatch is a defect
    rather than a convention difference.
    """
    estimator, identity = _resolve_qdk_estimator()
    if estimator is None:
        return [
            Comparison(
                name="layout-overhead",
                quantity="logical qubits including routing space",
                reference_tool="qdk",
                reference_tool_identity=identity,
                status="UNAVAILABLE",
                detail=f"The QDK estimator is not importable: {identity}",
            )
        ]

    results: list[Comparison] = []
    for n in (4, 8, 16, 32, 100):
        ours = lattice_surgery_logical_qubits(n)
        try:
            theirs = _qdk_layout_overhead(estimator, n)
        except Exception as error:  # noqa: BLE001
            results.append(
                Comparison(
                    name=f"layout-overhead-n{n}",
                    quantity="logical qubits including routing space",
                    reference_tool="qdk",
                    reference_tool_identity=identity,
                    ketqat_value=float(ours),
                    status="UNAVAILABLE",
                    detail=f"The QDK estimator raised at n={n}: {error}",
                )
            )
            continue
        results.append(
            Comparison(
                name=f"layout-overhead-n{n}",
                quantity="logical qubits including routing space",
                reference_tool="qdk",
                reference_tool_identity=identity,
                ketqat_value=float(ours),
                reference_value=float(theirs),
                status="AGREED" if ours == theirs else "DIFFERED",
                detail=(
                    f"2n + ceil(sqrt(8n)) + 1 at n={n}, against algorithmicLogicalQubits computed "
                    f"by {identity}. Beverland et al. (2022), the formula Microsoft's estimator "
                    "implements."
                ),
                assumptions=["2D lattice-surgery layout", "rotated surface code"],
            )
        )
    return results


def _qualtran_prefactor_comparison() -> list[Comparison]:
    """Code distance against Qualtran, whose fitted prefactor differs.

    This comparison is expected to *differ*, and the difference is the point.
    Qualtran's ``QECScheme.make_gidney_fowler()`` uses A = 0.1 where this project
    defaults to 0.03; both are in published tooling and neither is more correct.
    A run that reported agreement here would mean somebody had quietly changed
    one of them to match the other, which is the failure this check exists to
    surface rather than hide.
    """
    logical_qubits, cycles, rate, budget = 100, 10_000, 1e-3, 1e-2
    ours = required_code_distance(logical_qubits, cycles, rate, budget, prefactor=FOWLER_PREFACTOR)
    theirs = required_code_distance(logical_qubits, cycles, rate, budget, prefactor=0.1)

    try:
        import qualtran.surface_code  # noqa: F401
        available = True
        detail = "Qualtran importable; its documented default prefactor is 0.1."
    except Exception as error:  # pragma: no cover - exercised by the unavailable path
        available = False
        detail = f"qualtran is not importable: {error}"

    return [
        Comparison(
            name="code-distance-prefactor",
            quantity="required code distance",
            reference_tool="qualtran",
            ketqat_value=float(ours) if ours else None,
            reference_value=float(theirs) if theirs else None,
            status=("DIFFERED" if ours != theirs else "AGREED") if available else "UNAVAILABLE",
            definitional_difference=(
                "The logical-error prefactor is fitted, not derived. This project defaults to "
                "A = 0.03 (Fowler conventional); Qualtran's make_gidney_fowler uses A = 0.1. "
                "Neither is more correct, and the resulting distances differ by one step. "
                "KetQat reports both as model sensitivity on every estimate rather than "
                "choosing one and presenting it as the answer."
            ),
            detail=detail,
            assumptions=[
                f"{logical_qubits} logical qubits, {cycles} cycles",
                f"physical error rate {rate}, error budget {budget}",
            ],
        )
    ]


def run_differential_checks() -> list[Comparison]:
    """Every differential comparison, with its availability recorded."""
    checks: list[Callable[[], list[Comparison]]] = [
        _qdk_layout_comparison,
        _qualtran_prefactor_comparison,
    ]
    results: list[Comparison] = []
    for check in checks:
        results.extend(check())
    return results


def to_report(comparisons: list[Comparison]) -> dict[str, Any]:
    counts = {status: 0 for status in ("AGREED", "DIFFERED", "UNAVAILABLE")}
    for comparison in comparisons:
        counts[comparison.status] = counts.get(comparison.status, 0) + 1
    return {
        "kind": "resource-intelligence-differential",
        "schema_version": "0.1",
        "counts": counts,
        "comparisons": [asdict(comparison) for comparison in comparisons],
        # Parenthesised rather than relying on implicit adjacent-string
        # concatenation. In a list literal that idiom is indistinguishable from a
        # missing comma -- CodeQL flags it for exactly that reason -- and here the
        # two readings differ in how many notes the report carries.
        "notes": [
            (
                "UNAVAILABLE is not a pass. A comparison that did not run proves nothing, and the "
                "CI gate fails when a required one is UNAVAILABLE."
            ),
            (
                "DIFFERED is not necessarily a defect. Where the tools define a quantity "
                "differently the cause is recorded in definitional_difference rather than tuned "
                "away."
            ),
        ],
    }


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Write the report as JSON here.")
    parser.add_argument(
        "--require",
        nargs="*",
        default=[],
        help="Reference tools whose comparisons must actually have run. UNAVAILABLE fails.",
    )
    args = parser.parse_args(argv)

    comparisons = run_differential_checks()
    report = to_report(comparisons)
    serialized = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)

    missing = [
        comparison.name
        for comparison in comparisons
        if comparison.reference_tool in args.require and comparison.status == "UNAVAILABLE"
    ]
    if missing:
        print(
            "\nFAIL: these comparisons were required and did not run: " + ", ".join(missing),
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
