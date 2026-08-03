"""Compare independent QEC decoders on one shared, hashed sample.

**Sampling is direct shared-Stim sampling, not Sinter.** An earlier version of this
module claimed the decoders ran "through a single Sinter task collection"; they did not,
and the claim is corrected here rather than papered over. One detector/observable sample
is drawn once from Stim, its bytes are hashed, and every decoder adapter recomputes that
hash from the arrays it actually iterated -- so "identical samples" is proved by digest,
not inferred from equal shot counts, which any two independent runs of the same size
would also show.

**Separation is decided by paired per-shot inference, not interval overlap.** All
decoders see the same shots, so the comparison is paired: for each pair we count the
discordant shots (A wrong where B right, and conversely), run an exact McNemar binomial
test with Bonferroni correction across the family of pairs, and report a paired
risk-difference confidence interval. Marginal Wilson intervals are kept for scale, but
overlap between them does not decide anything -- on paired data the discordant counts can
settle what the marginals cannot.

**Abstention must not improve rank.** Coverage, conditional error (among decoded shots)
and unconditional risk (failure or abstention, over all shots) are reported separately,
and rank uses unconditional risk, so a decoder cannot climb the table by declining the
hard shots.

**Latency and throughput are different experiments.** Single-shot latency is measured
with warm-up and randomized-order repeats; throughput uses each library's supported batch
API. Both run in a spawned child process per decoder, so one decoder's allocations cannot
appear in another's numbers, and peak memory is the child's own `ru_maxrss` (whole
process RSS). Python-heap tracemalloc is *not* reported as memory.

A decoder that is not installed is recorded as **not run**, never a row of zeros.
`is_demo` and `publishable` are derived from what actually happened, not hardcoded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import queue as pyqueue
import platform
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

Z95 = 1.959963984540054


# --------------------------------------------------------------------------- sample


def sample_digest(detectors, observables) -> str:
    """SHA-256 over the exact sample bytes."""
    h = hashlib.sha256()
    h.update(detectors.tobytes())
    h.update(b"|")
    h.update(observables.tobytes())
    return h.hexdigest()


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


# --------------------------------------------------------------------------- stats


def wilson(errors: int, n: int, z: float = Z95) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = errors / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(b01: int, b10: int) -> float:
    """Two-sided McNemar p-value from the discordant counts.

    Exact binomial via integer arithmetic up to 1000 discordant shots -- `2.0**n`
    overflows a float past n ~ 1024, which is how the first version of this function
    died on real data. Beyond that, the continuity-corrected normal approximation,
    whose error at n > 1000 is far below anything that changes a decision at
    Bonferroni-corrected alpha.
    """
    n = b01 + b10
    if n == 0:
        return 1.0
    if n <= 1000:
        from fractions import Fraction

        k = min(b01, b10)
        tail = Fraction(sum(math.comb(n, i) for i in range(k + 1)), 2**n)
        return min(1.0, float(2 * tail))
    z = (abs(b01 - b10) - 1) / math.sqrt(n)
    return min(1.0, math.erfc(max(0.0, z) / math.sqrt(2.0)))


def paired_risk_difference(a_err: list[bool], b_err: list[bool]) -> dict[str, Any]:
    """Error-rate difference on the same shots, with a paired CI and McNemar test.

    Concordant shots carry no information about the *difference*, and the paired
    standard error correctly reflects that: it is driven by the discordant counts.
    """
    n = len(a_err)
    b01 = sum(1 for x, y in zip(a_err, b_err) if x and not y)  # A wrong, B right
    b10 = sum(1 for x, y in zip(a_err, b_err) if not x and y)  # A right, B wrong
    if n == 0:
        return {"n_paired": 0, "b01": 0, "b10": 0, "risk_difference": None, "ci_95": None, "mcnemar_p": None}
    d = (b01 - b10) / n
    se = math.sqrt(max(0.0, (b01 + b10) - (b01 - b10) ** 2 / n)) / n
    return {
        "n_paired": n,
        "b01": b01,
        "b10": b10,
        "risk_difference": d,
        "ci_95": (d - Z95 * se, d + Z95 * se),
        "mcnemar_p": mcnemar_exact_p(b01, b10),
    }


# --------------------------------------------------------------------------- adapters


@dataclass
class AccuracyResult:
    decoder: str
    library: str
    library_version: str
    available: bool
    config: dict[str, Any] = field(default_factory=dict)
    consumed_sample_sha256: str | None = None
    shots: int = 0
    decoded: int = 0
    abstentions: int = 0
    logical_errors: int = 0
    #: Per-shot outcome, "ok"/"err"/"abstain": the raw material for the paired stats.
    per_shot: list[str] = field(default_factory=list)
    not_run_reason: str | None = None


def _consume_and_score(result: AccuracyResult, detectors, observables, predict_one) -> None:
    """Iterate the shared sample, hashing each row as it is consumed.

    Rows of a C-contiguous boolean array concatenate to the array's bytes, so the
    running digest equals `sample_digest` exactly when every row was visited once, in
    order -- which is what "this decoder consumed the shared sample" means here.
    """
    digest = hashlib.sha256()
    for index in range(detectors.shape[0]):
        shot = detectors[index]
        digest.update(shot.tobytes())
        prediction = predict_one(shot)
        if prediction is None:
            result.per_shot.append("abstain")
        else:
            result.per_shot.append("err" if int(prediction) != int(observables[index, 0]) else "ok")
    digest.update(b"|")
    digest.update(observables.tobytes())
    result.consumed_sample_sha256 = digest.hexdigest()
    result.shots = len(result.per_shot)
    result.abstentions = sum(1 for s in result.per_shot if s == "abstain")
    result.decoded = result.shots - result.abstentions
    result.logical_errors = sum(1 for s in result.per_shot if s == "err")


def _accuracy_pymatching(dem, detectors, observables) -> AccuracyResult:
    import pymatching

    matching = pymatching.Matching.from_detector_error_model(dem)
    result = AccuracyResult(
        "pymatching-mwpm", "pymatching", pymatching.__version__, True,
        config={"algorithm": "minimum-weight perfect matching", "construction": "from_detector_error_model"},
    )
    _consume_and_score(result, detectors, observables, lambda shot: int(matching.decode(shot)[0]))
    return result


def _accuracy_beliefmatching(dem, detectors, observables) -> AccuracyResult:
    import beliefmatching
    from beliefmatching import BeliefMatching

    decoder = BeliefMatching.from_detector_error_model(dem)
    result = AccuracyResult(
        "beliefmatching", "beliefmatching",
        getattr(beliefmatching, "__version__", "unknown"), True,
        config={"algorithm": "belief propagation + MWPM", "construction": "from_detector_error_model"},
    )
    _consume_and_score(result, detectors, observables, lambda shot: int(decoder.decode(shot)[0]))
    return result


def _accuracy_tesseract(dem, detectors, observables) -> AccuracyResult:
    import numpy as np
    import tesseract_decoder

    decoders = tesseract_decoder.make_tesseract_sinter_decoders_dict()
    name, sinter_decoder = next(iter(decoders.items()))
    compiled = sinter_decoder.compile_decoder_for_dem(dem=dem)
    result = AccuracyResult(
        "tesseract", "tesseract-decoder",
        getattr(tesseract_decoder, "__version__", "unknown"), True,
        config={"algorithm": "tesseract most-likely-error", "sinter_decoder_key": name},
    )

    def predict(shot):
        packed = np.packbits(shot, bitorder="little").reshape(1, -1)
        out = compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        return int(np.unpackbits(out, bitorder="little")[0])

    _consume_and_score(result, detectors, observables, predict)
    return result


ACCURACY_ADAPTERS: dict[str, Callable] = {
    "pymatching-mwpm": _accuracy_pymatching,
    "beliefmatching": _accuracy_beliefmatching,
    "tesseract": _accuracy_tesseract,
}


# ----------------------------------------------------------------- timing (isolated)


def _timing_child(name: str, dem_text: str, det_bytes: bytes, det_shape,
                  latency_shots: int, repeats: int, seed: int, queue) -> None:
    """One decoder, in its own spawned process: its own caches, its own RSS."""
    try:
        # Bound the child's address space where the platform honors it
        # (Linux). macOS largely ignores RLIMIT_AS, so the parent's
        # wall-clock bound is the enforcement that always works; this is
        # defense in depth, not the primary guard.
        try:
            resource.setrlimit(resource.RLIMIT_AS, (TIMING_CHILD_MEMORY_BOUND_BYTES, TIMING_CHILD_MEMORY_BOUND_BYTES))
        except (ValueError, OSError):
            # macOS refuses or ignores RLIMIT_AS; the parent's wall-clock
            # bound is the enforcement that works everywhere, so a platform
            # that rejects the limit is degraded, not broken.
            pass
        import numpy as np
        import stim

        dem = stim.DetectorErrorModel(dem_text)
        detectors = np.frombuffer(det_bytes, dtype=np.bool_).reshape(det_shape)

        if name == "pymatching-mwpm":
            import pymatching
            started = time.perf_counter()
            decoder = pymatching.Matching.from_detector_error_model(dem)
            compile_seconds = time.perf_counter() - started
            one = lambda shot: decoder.decode(shot)  # noqa: E731
            batch = lambda: decoder.decode_batch(detectors)  # noqa: E731
            batch_api = "pymatching.Matching.decode_batch"
        elif name == "beliefmatching":
            from beliefmatching import BeliefMatching
            started = time.perf_counter()
            decoder = BeliefMatching.from_detector_error_model(dem)
            compile_seconds = time.perf_counter() - started
            one = lambda shot: decoder.decode(shot)  # noqa: E731
            batch = lambda: decoder.decode_batch(detectors)  # noqa: E731
            batch_api = "beliefmatching.BeliefMatching.decode_batch"
        elif name == "tesseract":
            import tesseract_decoder
            started = time.perf_counter()
            sinter_decoder = next(iter(tesseract_decoder.make_tesseract_sinter_decoders_dict().values()))
            compiled = sinter_decoder.compile_decoder_for_dem(dem=dem)
            compile_seconds = time.perf_counter() - started
            packed_all = np.packbits(detectors, axis=1, bitorder="little")
            one = lambda shot: compiled.decode_shots_bit_packed(  # noqa: E731
                bit_packed_detection_event_data=np.packbits(shot, bitorder="little").reshape(1, -1))
            batch = lambda: compiled.decode_shots_bit_packed(  # noqa: E731
                bit_packed_detection_event_data=packed_all)
            batch_api = "tesseract compiled.decode_shots_bit_packed (whole sample)"
        else:
            queue.put({"error": f"unknown decoder {name}"})
            return

        # Warm-up: first calls pay one-time costs that are not per-shot latency.
        for index in range(min(50, detectors.shape[0])):
            one(detectors[index])

        rng = random.Random(seed)
        indices = list(range(min(latency_shots, detectors.shape[0])))
        latencies_us: list[float] = []
        for _ in range(repeats):
            rng.shuffle(indices)  # randomized order per repeat
            for index in indices:
                shot = detectors[index]
                t0 = time.perf_counter()
                one(shot)
                latencies_us.append((time.perf_counter() - t0) * 1e6)

        batch_times: list[float] = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            batch()
            batch_times.append(time.perf_counter() - t0)

        latencies_us.sort()
        pct = lambda f: latencies_us[min(len(latencies_us) - 1, int(f * (len(latencies_us) - 1)))]  # noqa: E731
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else rss * 1024  # macOS bytes, Linux KB
        median_batch = sorted(batch_times)[len(batch_times) // 2]
        queue.put({
            "compile_seconds": compile_seconds,
            "latency_measured_shots": len(indices),
            "latency_repeats": repeats,
            "latency_p50_us": pct(0.50),
            "latency_p95_us": pct(0.95),
            "latency_p99_us": pct(0.99),
            "batch_api": batch_api,
            "batch_repeats": repeats,
            "batch_seconds_median": median_batch,
            "batch_throughput_shots_per_second": detectors.shape[0] / median_batch,
            "peak_rss_bytes": int(rss_bytes),
            "isolation": "spawned child process per decoder",
        })
    except Exception as exc:  # noqa: BLE001 - the parent records the failure
        queue.put({"error": f"{type(exc).__name__}: {exc}"})


TIMING_WALL_CLOCK_BOUND_SECONDS = 600
TIMING_CHILD_MEMORY_BOUND_BYTES = 8 * 1024**3


def _timing_failure(name: str, error_class: str, *, elapsed: float,
                    exitcode: int | None = None) -> dict[str, Any]:
    """Structured failed-cell record (ketqat-benchmarks#9).

    A timing cell that could not be measured is data about the decoder --
    "tesseract exhausts memory here" is a finding -- so the failure carries
    everything a reader needs to classify it: which bound fired, how the
    child died, and the environment it died in. Downstream datasets copy
    this record instead of leaving an unexplained hole.
    """
    signal_number = -exitcode if exitcode is not None and exitcode < 0 else None
    record: dict[str, Any] = {
        "error": error_class,
        "error_class": error_class,
        "decoder": name,
        "elapsed_seconds": round(elapsed, 3),
        "wall_clock_bound_seconds": TIMING_WALL_CLOCK_BOUND_SECONDS,
        "memory_bound_bytes": TIMING_CHILD_MEMORY_BOUND_BYTES,
        "exitcode": exitcode,
        "signal": signal_number,
        "suspected_oom": signal_number == 9,
        "environment": machine_metadata(),
    }
    return record


def timing_for(name: str, dem, detectors, latency_shots: int, repeats: int, seed: int) -> dict[str, Any]:
    ctx = multiprocessing.get_context("spawn")
    queue: Any = ctx.Queue()
    proc = ctx.Process(
        target=_timing_child,
        args=(name, str(dem), detectors.tobytes(), detectors.shape, latency_shots, repeats, seed, queue),
    )
    proc.start()
    # Poll instead of a single long queue.get: a child that dies without
    # posting must be detected within a second, not after a 30-minute
    # timeout with the parent at 0% CPU -- which is exactly how the d=7
    # grid run wedged for over an hour (ketqat-benchmarks#9).
    start = time.perf_counter()
    payload: dict[str, Any] | None = None
    while True:
        try:
            payload = queue.get(timeout=1.0)
            break
        except pyqueue.Empty:
            # No payload within this 1s tick; fall through to the liveness
            # and wall-clock checks below, which are the point of polling.
            pass
        elapsed = time.perf_counter() - start
        if not proc.is_alive():
            # One last drain: the child may have posted between the timeout
            # and the liveness check.
            try:
                payload = queue.get(timeout=1.0)
                break
            except pyqueue.Empty:
                payload = _timing_failure(name, "child-died", elapsed=elapsed, exitcode=proc.exitcode)
                break
        if elapsed > TIMING_WALL_CLOCK_BOUND_SECONDS:
            proc.terminate()
            proc.join(timeout=10)
            if proc.is_alive():
                proc.kill()
            payload = _timing_failure(name, "timeout", elapsed=elapsed, exitcode=proc.exitcode)
            break
    proc.join(timeout=60)
    if proc.is_alive():
        proc.kill()
        proc.join(timeout=10)
    return payload


# --------------------------------------------------------------------------- report


def machine_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
    }


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def run_comparison(distance: int, rounds: int, noise: float, shots: int, seed: int = 7,
                   latency_shots: int = 1000, timing_repeats: int = 3,
                   with_timing: bool = True) -> dict[str, Any]:
    import stim

    circuit = build_circuit(distance, rounds, noise)
    dem = circuit.detector_error_model(decompose_errors=True)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots, separate_observables=True)
    master_hash = sample_digest(detectors, observables)

    results: list[AccuracyResult] = []
    for name, adapter in ACCURACY_ADAPTERS.items():
        try:
            outcome = adapter(dem, detectors, observables)
            if outcome.consumed_sample_sha256 != master_hash:
                # A decoder that did not provably consume the shared sample is excluded
                # from the paired comparison rather than compared on faith.
                outcome.available = False
                outcome.not_run_reason = (
                    f"consumed-sample hash {outcome.consumed_sample_sha256[:12]} != master {master_hash[:12]}"
                )
            results.append(outcome)
        except Exception as exc:  # noqa: BLE001
            results.append(AccuracyResult(name, name, "", False, not_run_reason=f"{type(exc).__name__}: {exc}"))

    ran = [r for r in results if r.available]
    timings: dict[str, dict[str, Any]] = {}
    if with_timing:
        for r in ran:
            timing = timing_for(r.decoder, dem, detectors, latency_shots, timing_repeats, seed)
            if "error_class" in timing:
                # The cell context lives here, not in timing_for: a failed
                # cell must name what was being measured when it failed.
                timing.update({"distance": distance, "rounds": rounds, "noise": noise, "seed": seed})
            timings[r.decoder] = timing

    # Paired inference over every pair, Bonferroni-corrected across the family.
    pairs: list[dict[str, Any]] = []
    alpha = 0.05
    n_pairs = max(1, len(ran) * (len(ran) - 1) // 2)
    for i in range(len(ran)):
        for j in range(i + 1, len(ran)):
            a, b = ran[i], ran[j]
            mask = [sa != "abstain" and sb != "abstain" for sa, sb in zip(a.per_shot, b.per_shot)]
            a_err = [s == "err" for s, m in zip(a.per_shot, mask) if m]
            b_err = [s == "err" for s, m in zip(b.per_shot, mask) if m]
            stats = paired_risk_difference(a_err, b_err)
            stats.update({
                "a": a.decoder,
                "b": b.decoder,
                "alpha_bonferroni": alpha / n_pairs,
                "significant_after_bonferroni": (
                    stats["mcnemar_p"] is not None and stats["mcnemar_p"] < alpha / n_pairs
                ),
            })
            pairs.append(stats)

    decoder_rows = []
    for r in results:
        row = asdict(r)
        # Kept, compactly: the raw per-shot outcomes are the material for reanalysis.
        row["per_shot"] = "".join({"ok": ".", "err": "E", "abstain": "a"}[s] for s in r.per_shot)
        row["coverage"] = (r.decoded / r.shots) if r.shots else None
        row["conditional_error_rate"] = (r.logical_errors / r.decoded) if r.decoded else None
        row["unconditional_risk"] = ((r.logical_errors + r.abstentions) / r.shots) if r.shots else None
        row["wilson_95_conditional"] = wilson(r.logical_errors, r.decoded)
        row["timing"] = timings.get(r.decoder)
        decoder_rows.append(row)

    # Rank by unconditional risk, so abstention cannot improve position.
    ranked = sorted((row for row in decoder_rows if row["available"]),
                    key=lambda row: row["unconditional_risk"])
    for position, row in enumerate(ranked, start=1):
        row["rank_by_unconditional_risk"] = position

    installed = {}
    for module_name in ("stim", "pymatching", "beliefmatching", "tesseract_decoder", "sinter", "numpy"):
        try:
            module = __import__(module_name)
            installed[module_name] = getattr(module, "__version__", "unknown")
        except ImportError:
            installed[module_name] = None

    report: dict[str, Any] = {
        "schema_version": "0.2",
        "supersedes": "results/d3-r3-p02.json (schema 0.1, commit 8545166)",
        "execution_class": "SIMULATION",
        "experiment": {
            "code": "surface_code:rotated_memory_z",
            "distance": distance, "rounds": rounds, "noise": noise,
            "shots": shots, "seed": seed,
            "sampling": "direct shared-Stim detector sampling (not Sinter)",
            "circuit_sha256": hashlib.sha256(str(circuit).encode()).hexdigest(),
            "dem_sha256": hashlib.sha256(str(dem).encode()).hexdigest(),
            "sample_sha256": master_hash,
        },
        "provenance": {
            "machine": machine_metadata(),
            "installed_versions": installed,
            "stim_version": stim.__version__,
            "latency_protocol": {
                "warmup_shots": 50, "measured_shots": latency_shots,
                "repeats": timing_repeats, "order": "randomized per repeat",
                "isolation": "spawned child process per decoder",
            },
            "memory_note": "peak_rss_bytes is child-process ru_maxrss (whole-process RSS); tracemalloc is not used",
        },
        "decoders": decoder_rows,
        "paired_comparisons": pairs,
        "decoders_run": len(ran),
        "is_leaderboard": len(ran) >= 2,
    }
    # Derived, never asserted.
    report["is_demo"] = len(ran) == 0
    report["publishable"] = (
        len(ran) >= 2
        and not report["is_demo"]
        and all(row["consumed_sample_sha256"] == master_hash for row in decoder_rows if row["available"])
    )
    report["reproducibility_sha256"] = hashlib.sha256(_canonical(report).encode()).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare QEC decoders on one shared, hashed sample.")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--max-shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--latency-shots", type=int, default=1000)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--no-timing", action="store_true", help="Accuracy and paired stats only.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = run_comparison(args.distance, args.rounds, args.noise, args.max_shots, args.seed,
                            args.latency_shots, args.timing_repeats, with_timing=not args.no_timing)
    text = json.dumps(report, indent=2, default=str)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"Wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
