"""The measured classical baseline is a fact, so it is checked like one.

These assert properties of the *measurement*, not of a stored number. A test
pinning "12 qubits takes 14ms" would fail on every machine but the one that
wrote it, which teaches the next person to delete the test rather than trust it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ketqat_benchmarks.classical_baseline import measure, record

EVIDENCE = Path(__file__).resolve().parents[1] / "results" / "reference" / "classical-baseline.json"


def test_simulation_cost_grows_with_qubit_count():
    """Exact simulation is exponential in qubits. If it were not, it would not be exact."""
    small = measure(6, 2, repeats=2)
    large = measure(10, 2, repeats=2)
    assert large.runtime_seconds > small.runtime_seconds
    # The statevector itself is the hard floor, and it is exact arithmetic.
    assert large.statevector_bytes == (1 << 10) * 16
    assert large.statevector_bytes == small.statevector_bytes * (1 << 4)


def test_the_reported_runtime_is_the_minimum_not_an_average():
    """A mean drags in whatever else the machine was doing."""
    result = measure(8, 2, repeats=4)
    assert result.runtime_seconds == min(result.runtime_all_seconds)
    assert result.runtime_seconds <= result.runtime_median_seconds
    assert len(result.runtime_all_seconds) == 4


def test_every_repeat_is_retained_so_the_spread_stays_visible():
    result = measure(6, 1, repeats=3)
    assert len(result.runtime_all_seconds) == 3
    assert all(value > 0 for value in result.runtime_all_seconds)


def test_the_environment_travels_with_the_measurement():
    """A runtime without its machine and date is not reproducible.

    These are exactly the fields the SDK's ClassicalBaseline marks required, so
    a measurement that passes here can become a baseline without anything being
    invented to fill a gap.
    """
    evidence = record([(6, 1)], repeats=2)
    assert evidence.measured_on
    assert evidence.command.startswith("python -m ketqat_benchmarks.classical_baseline")
    for key in ("cpu", "platform", "python_version", "python_implementation", "cores_used"):
        assert evidence.environment.get(key), f"environment is missing {key}"
    assert evidence.kind == "MEASURED_CLASSICAL_BASELINE"


def test_the_measurement_states_what_it_understates():
    """The bias direction is recorded because it runs against this project's interest.

    An unoptimised simulator understates classical capability, which flatters
    the quantum side. A reference case that hides that is worth less than no
    reference case.
    """
    evidence = record([(6, 1)], repeats=1)
    joined = " ".join(evidence.limitations).lower()
    assert "understates classical" in joined
    assert "not evidence about any organisation" in joined


def test_the_committed_evidence_file_is_a_real_dated_measurement():
    """The file the product ships against, checked rather than assumed."""
    assert EVIDENCE.exists(), "the reference measurement has not been recorded"
    data = json.loads(EVIDENCE.read_text())

    assert data["kind"] == "MEASURED_CLASSICAL_BASELINE"
    assert data["measured_on"], "an undated measurement gets quoted after it stops being true"
    assert data["environment"]["cpu"] not in ("", "unknown"), "the CPU is how a reader judges transfer"
    assert data["measurements"], "no measurements recorded"

    for entry in data["measurements"]:
        assert entry["runtime_seconds"] > 0
        assert entry["runtime_seconds"] == min(entry["runtime_all_seconds"])
        assert entry["gate_count"] > 0

    # Monotone in problem size, which is the property that makes it a baseline
    # rather than a single anecdote.
    runtimes = [entry["runtime_seconds"] for entry in data["measurements"]]
    assert runtimes == sorted(runtimes), "a larger circuit must not simulate faster"


def test_the_evidence_carries_no_quantum_advantage_claim():
    data = json.loads(EVIDENCE.read_text())
    blob = json.dumps(data).lower()
    for forbidden in ("advantage", "supremacy", "outperform", "faster than quantum"):
        assert forbidden not in blob, f"the measurement must claim nothing about advantage: {forbidden}"
