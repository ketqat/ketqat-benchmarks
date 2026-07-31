"""The three-decoder gate, inside the package rather than beside it (#6).

A comparison that quietly drops to one decoder, or whose entrants did not provably
consume the shared sample, still emits a valid-looking report. This refuses it.

It lived in `scripts/` and so was not in the wheel: a clean-room user could produce a
comparison and had no way to check it, which is the half of the tool that matters. It is
now importable and installed as the `ketqat-benchmarks-gate` console script;
`scripts/assert_three_decoders.py` is a shim so CI and existing instructions still work.

Nothing here asserts `is_demo` or `publishable` -- it checks what the library *derived*.
Asserting them would let a report pass by claiming to be publishable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def check(report: dict[str, Any]) -> list[str]:
    """Every reason this report must not be published. Empty means it may be."""
    failures: list[str] = []

    ran = [d for d in report["decoders"] if d["available"]]
    missing = [(d["decoder"], d["not_run_reason"]) for d in report["decoders"] if not d["available"]]
    if len(ran) < 3:
        failures.append(f"{len(ran)} of 3 decoders ran; missing: {missing}")

    # The shared sample is what makes the comparison paired. A decoder that cannot show
    # it consumed the same shots is not a competitor in this table.
    master = report["experiment"]["sample_sha256"]
    for decoder in ran:
        if decoder["consumed_sample_sha256"] != master:
            failures.append(f"{decoder['decoder']} did not provably consume the shared sample")

    if report.get("is_demo") is not False:
        failures.append("report derived is_demo != False (nothing real ran?)")
    if report.get("publishable") is not True:
        failures.append("report is not publishable by its own derivation")
    if not report.get("paired_comparisons"):
        failures.append("no paired comparisons present")

    # This harness samples directly from Stim. A report claiming Sinter would be
    # describing a provenance it does not have.
    sampling = report["experiment"]["sampling"]
    if "Sinter" in sampling and "not Sinter" not in sampling:
        failures.append("sampling claims Sinter; this harness samples directly from Stim")

    return failures


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        print("usage: ketqat-benchmarks-gate <comparison.json>", file=sys.stderr)
        return 2

    report = json.loads(Path(arguments[0]).read_text(encoding="utf-8"))
    failures = check(report)

    if failures:
        print("FAIL:")
        for failure in failures:
            print("  -", failure)
        return 1

    ran = [d for d in report["decoders"] if d["available"]]
    print(
        f"PASS: {len(ran)} decoders, sample {report['experiment']['sample_sha256'][:12]}, "
        f"publishable derived True, {len(report['paired_comparisons'])} paired comparisons"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
