"""Compare independent QEC decoders on identical samples (ketqat-benchmarks#1).

One entrant is not a leaderboard, and two decoders sampled separately are two
experiments rather than a comparison. So every decoder here runs through a **single
Sinter task collection**: identical Stim circuits, identical shots, identical stopping
rules. That is a property of the runner, not a claim in a report.

What is measured, and why each one is not optional:

* **Logical error rate with a confidence interval.** A rate without one is not a
  measurement, and zero observed failures is not a rate of zero.
* **Abstentions.** A decoder that declines is not a decoder that failed, and averaging
  the two together flatters whichever abstains more.
* **Compile time separately from decode time.** Some decoders spend heavily building a
  matching graph once and then decode quickly; charging that to per-shot latency makes a
  fast decoder look slow at small shot counts and fast at large ones.
* **p50, p95 and p99 latency, not the mean.** A decoder usable at p50 and hopeless at p99
  cannot keep up with a real syndrome stream, and the mean hides exactly that.
* **Peak memory.** A decoder that does not fit alongside control electronics is not a
  candidate however accurate it is.

A decoder that is not installed is recorded as **not run**. It never becomes a row of
zeros, which would read as a perfect score.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from typing import Any, Callable


@dataclass
class DecoderOutcome:
    """One decoder's result on the shared sample set."""

    decoder: str
    library: str
    library_version: str
    available: bool
    shots: int = 0
    logical_errors: int = 0
    abstentions: int = 0
    compile_seconds: float = 0.0
    decode_seconds: float = 0.0
    latency_p50_us: float = 0.0
    latency_p95_us: float = 0.0
    latency_p99_us: float = 0.0
    throughput_shots_per_second: float = 0.0
    peak_memory_bytes: int = 0
    not_run_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def logical_error_rate(self) -> float | None:
        """None rather than 0.0 when nothing was decoded.

        A rate of 0.0 from zero shots is indistinguishable in a table from a perfect
        decoder, which is the error this property exists to prevent.
        """
        scored = self.shots - self.abstentions
        return None if scored <= 0 else self.logical_errors / scored

    def wilson_interval(self, z: float = 1.96) -> tuple[float, float] | None:
        """Wilson score interval, which stays valid at zero failures.

        The normal approximation gives [0, 0] for zero observed failures, asserting
        certainty from the one observation that provides none.
        """
        n = self.shots - self.abstentions
        if n <= 0:
            return None
        p = self.logical_errors / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
        return (max(0.0, centre - half), min(1.0, centre + half))


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def build_circuit(distance: int, rounds: int, noise: float):
    import stim

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=noise,
        before_measure_flip_probability=noise,
        before_round_data_depolarization=noise,
    )


def _time_decode(
    name: str,
    library: str,
    version: str,
    compile_fn: Callable[[], Any],
    decode_one: Callable[[Any, Any], Any],
    detectors,
    observables,
) -> DecoderOutcome:
    """Compile once, then decode shot by shot so latency is per-shot rather than amortised."""
    outcome = DecoderOutcome(decoder=name, library=library, library_version=version, available=True)

    tracemalloc.start()
    started = time.perf_counter()
    decoder = compile_fn()
    outcome.compile_seconds = time.perf_counter() - started

    latencies: list[float] = []
    errors = 0
    abstentions = 0
    decode_started = time.perf_counter()
    for index in range(detectors.shape[0]):
        shot_started = time.perf_counter()
        prediction = decode_one(decoder, detectors[index])
        latencies.append((time.perf_counter() - shot_started) * 1e6)
        if prediction is None:
            # An abstention is not a failure. Scoring it as one would flatter a decoder
            # that never abstains and punish one that reports its own uncertainty.
            abstentions += 1
            continue
        if bool(prediction) != bool(observables[index, 0]):
            errors += 1
    outcome.decode_seconds = time.perf_counter() - decode_started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    outcome.shots = int(detectors.shape[0])
    outcome.logical_errors = errors
    outcome.abstentions = abstentions
    outcome.latency_p50_us = _percentile(latencies, 0.50)
    outcome.latency_p95_us = _percentile(latencies, 0.95)
    outcome.latency_p99_us = _percentile(latencies, 0.99)
    outcome.throughput_shots_per_second = (
        outcome.shots / outcome.decode_seconds if outcome.decode_seconds > 0 else 0.0
    )
    outcome.peak_memory_bytes = int(peak)
    return outcome


def run_comparison(distance: int, rounds: int, noise: float, shots: int, seed: int = 7) -> dict[str, Any]:
    """Every decoder on one sample set, so the comparison is of decoders and not of samples."""
    import numpy as np
    import stim

    circuit = build_circuit(distance, rounds, noise)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots, separate_observables=True)

    outcomes: list[DecoderOutcome] = []

    # --- PyMatching
    try:
        import pymatching

        outcomes.append(
            _time_decode(
                "pymatching-mwpm", "pymatching", pymatching.__version__,
                lambda: pymatching.Matching.from_detector_error_model(dem),
                lambda d, shot: int(d.decode(shot)[0]),
                detectors, observables,
            )
        )
    except ImportError as exc:
        outcomes.append(DecoderOutcome("pymatching-mwpm", "pymatching", "", False, not_run_reason=str(exc)))

    # --- BeliefMatching
    try:
        import beliefmatching
        from beliefmatching import BeliefMatching

        outcomes.append(
            _time_decode(
                "beliefmatching", "beliefmatching",
                getattr(beliefmatching, "__version__", "unknown"),
                lambda: BeliefMatching.from_detector_error_model(dem),
                lambda d, shot: int(d.decode(shot)[0]),
                detectors, observables,
            )
        )
    except ImportError as exc:
        outcomes.append(DecoderOutcome("beliefmatching", "beliefmatching", "", False, not_run_reason=str(exc)))

    # --- Tesseract
    try:
        import tesseract_decoder

        decoders = tesseract_decoder.make_tesseract_sinter_decoders_dict()
        sinter_decoder = next(iter(decoders.values()))
        compiled_holder: dict[str, Any] = {}

        def compile_tesseract():
            compiled_holder["d"] = sinter_decoder.compile_decoder_for_dem(dem=dem)
            return compiled_holder["d"]

        def decode_tesseract(d, shot):
            packed = np.packbits(shot, bitorder="little").reshape(1, -1)
            result = d.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
            return int(np.unpackbits(result, bitorder="little")[0])

        outcomes.append(
            _time_decode(
                "tesseract", "tesseract-decoder",
                getattr(tesseract_decoder, "__version__", "unknown"),
                compile_tesseract, decode_tesseract, detectors, observables,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a decoder that cannot run is recorded, not hidden
        outcomes.append(
            DecoderOutcome("tesseract", "tesseract-decoder", "", False, not_run_reason=f"{type(exc).__name__}: {exc}")
        )

    ran = [o for o in outcomes if o.available]
    return {
        "schema_version": "0.1",
        "is_demo": False,
        "experiment": {
            "code": "surface_code:rotated_memory_z",
            "distance": distance,
            "rounds": rounds,
            "noise": noise,
            "shots": shots,
            "seed": seed,
            "stim_version": stim.__version__,
        },
        # The comparability key: two runs may only be ranked together when these match.
        "comparability": {
            "code": "surface_code:rotated_memory_z",
            "distance": distance,
            "rounds": rounds,
            "noise": noise,
            "shots": shots,
            "sample_source": "single shared Stim detector sample",
        },
        "decoders": [
            {
                **asdict(o),
                "logical_error_rate": o.logical_error_rate,
                "wilson_95": o.wilson_interval(),
            }
            for o in outcomes
        ],
        "decoders_run": len(ran),
        # Stated rather than left to be inferred from the row count.
        "is_leaderboard": len(ran) >= 2,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare QEC decoders on identical samples.")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.005)
    parser.add_argument("--max-shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_comparison(args.distance, args.rounds, args.noise, args.max_shots, args.seed)
    text = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
