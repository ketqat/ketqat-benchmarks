"""Measure a classical baseline, so a KetQat assessment can compare against a fact.

Every number KetQat Intelligence produces today is correct and none of it is
about anything real. The economic half of an assessment needs a *measured*
classical runtime, and until one exists the product can only demonstrate its
own arithmetic (ketqat-benchmarks#15, ketqat-planning#124).

## What this measures, and what it does not

The workload is **exact simulation of a quantum circuit on a classical
machine**: build the statevector, apply each gate, read the output
distribution. It is chosen because it is the one classical task whose quantum
counterpart is unambiguous -- the same circuit, run on hardware -- so the
comparison needs no mapping assumption between two different algorithms.

It is emphatically **not** a claim about quantum advantage. Simulating a small
circuit classically is cheap, and the honest conclusion for every size this
runs at is that classical wins by a wide margin. That is the point: a reference
case whose measured answer is "do not use a quantum computer for this" is worth
more than a fixture engineered to look favourable.

## Why the measurement is shaped this way

**Wall clock, repeated, minimum reported.** The minimum over repeats is the
figure least contaminated by other work on the machine -- a mean drags in
whatever else the OS was doing, and a decision maker reading "1.2 s" should get
the machine's capability, not its background load. The full distribution is
recorded anyway so the spread is visible.

**The environment travels with the number.** A runtime without the CPU, the
Python version, the date and the exact command is not reproducible and will be
quoted years after it stopped being true. Every field the SDK's
`ClassicalBaseline` marks required is captured here.

**No dependency on the SDK.** This module imports nothing from `ketqat-sdk`,
so the measurement cannot accidentally be shaped by the thing it is meant to
check. It emits a plain JSON evidence file; converting that into a
`ClassicalBaseline` is a separate, explicit step.
"""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

#: Repeats per size. Enough for a stable minimum without turning a reference
#: measurement into a benchmark run.
DEFAULT_REPEATS = 5


@dataclass
class Measurement:
    qubits: int
    gate_count: int
    #: Seconds. The minimum over repeats -- see the module note.
    runtime_seconds: float
    runtime_all_seconds: list[float]
    runtime_median_seconds: float
    #: Bytes held by the statevector alone, 2^n complex128.
    statevector_bytes: int


@dataclass
class BaselineEvidence:
    kind: str
    schema_version: str
    workload: str
    description: str
    measured_on: str
    command: str
    environment: dict[str, Any]
    measurements: list[Measurement]
    limitations: list[str] = field(default_factory=list)


def _statevector_simulate(qubits: int, layers: int) -> tuple[int, list[complex]]:
    """A deliberately plain statevector simulation.

    Written out rather than delegated to a library so the thing being timed is
    visible in this file. It is not the fastest possible simulator, and the
    evidence file says so -- a baseline measured with an unoptimised
    implementation understates what classical hardware can do, which biases
    *against* classical and so against the conclusion this project would prefer
    to draw. Recording that direction matters more than closing the gap.
    """
    size = 1 << qubits
    state = [0j] * size
    state[0] = 1 + 0j
    gates = 0

    inv_sqrt2 = 1 / math.sqrt(2)

    def apply_h(target: int) -> None:
        stride = 1 << target
        for base in range(0, size, stride << 1):
            for offset in range(base, base + stride):
                a = state[offset]
                b = state[offset + stride]
                state[offset] = (a + b) * inv_sqrt2
                state[offset + stride] = (a - b) * inv_sqrt2

    def apply_cx(control: int, target: int) -> None:
        cbit = 1 << control
        tbit = 1 << target
        for index in range(size):
            if index & cbit and not index & tbit:
                partner = index | tbit
                state[index], state[partner] = state[partner], state[index]

    def apply_t(target: int) -> None:
        phase = complex(inv_sqrt2, inv_sqrt2)
        tbit = 1 << target
        for index in range(size):
            if index & tbit:
                state[index] *= phase

    for layer in range(layers):
        for qubit in range(qubits):
            apply_h(qubit)
            gates += 1
        for qubit in range(qubits - 1):
            apply_cx(qubit, qubit + 1)
            gates += 1
        for qubit in range(qubits):
            if (qubit + layer) % 2 == 0:
                apply_t(qubit)
                gates += 1

    return gates, state


def measure(qubits: int, layers: int, repeats: int = DEFAULT_REPEATS) -> Measurement:
    timings: list[float] = []
    gate_count = 0
    for _ in range(repeats):
        start = time.perf_counter()
        gate_count, _state = _statevector_simulate(qubits, layers)
        timings.append(time.perf_counter() - start)
    return Measurement(
        qubits=qubits,
        gate_count=gate_count,
        runtime_seconds=min(timings),
        runtime_all_seconds=timings,
        runtime_median_seconds=statistics.median(timings),
        # 2^n amplitudes, each a complex128.
        statevector_bytes=(1 << qubits) * 16,
    )


def _cpu_model() -> str:
    """The CPU, by whatever the platform will tell us.

    Falls back to `platform.processor()` rather than guessing. An unknown CPU is
    recorded as unknown; a plausible-looking wrong one would be worse, because
    it is the field a reader uses to judge whether the number transfers to their
    machine.
    """
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5, check=True,
            ).stdout.strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:  # pragma: no cover - environment dependent
        pass
    return platform.processor() or "unknown"


def record(sizes: list[tuple[int, int]], repeats: int = DEFAULT_REPEATS) -> BaselineEvidence:
    measurements = [measure(qubits, layers, repeats) for qubits, layers in sizes]
    sizes_argument = " ".join(f"{q}x{l}" for q, l in sizes)
    return BaselineEvidence(
        kind="MEASURED_CLASSICAL_BASELINE",
        schema_version="0.1",
        workload="Exact statevector simulation of a layered H / CX / T circuit",
        description=(
            "Wall-clock time to simulate the circuit exactly on one CPU core, in pure Python. "
            "The minimum over repeats is reported; the full set is retained so the spread is visible."
        ),
        measured_on=date.today().isoformat(),
        command=(
            f"python -m ketqat_benchmarks.classical_baseline --sizes {sizes_argument} --repeats {repeats}"
        ),
        environment={
            "cpu": _cpu_model(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "cores_used": 1,
        },
        measurements=measurements,
        # Parenthesised rather than relying on implicit adjacent-string
        # concatenation: in a list literal that idiom is indistinguishable from
        # a missing comma, and here the two readings differ in how many
        # limitations the evidence file claims. CodeQL flags it for that reason.
        limitations=[
            (
                "Single-core pure Python. An optimised simulator on the same machine would be far faster, "
                "so this understates classical capability -- a bias against classical, and therefore against "
                "the conclusion this project would otherwise prefer to draw."
            ),
            (
                "Exact simulation only. Approximate methods (tensor networks, Clifford+T decompositions) can "
                "handle much larger circuits and are not measured here."
            ),
            (
                "One machine, one date. A runtime is a property of the hardware it ran on and transfers to "
                "another machine only as an order of magnitude."
            ),
            (
                "This is a reference measurement for demonstrating the assessment pipeline. It is not evidence "
                "about any organisation's production workload."
            ),        ],
    )


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="*", default=["8x2", "10x2", "12x2"],
        help="QUBITSxLAYERS pairs, for example 12x2.",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--output", help="Write the evidence file here.")
    args = parser.parse_args(argv)

    sizes: list[tuple[int, int]] = []
    for entry in args.sizes:
        qubits, _, layers = entry.partition("x")
        sizes.append((int(qubits), int(layers or 1)))

    evidence = record(sizes, args.repeats)
    serialized = json.dumps(asdict(evidence), indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(serialized + "\n")
    print(serialized)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
