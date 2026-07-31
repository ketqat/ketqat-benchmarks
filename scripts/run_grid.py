#!/usr/bin/env python3
"""Benchmark grid: multiple distances, rounds, error rates and seeds.

One d=3 point is not a characterization -- decoder ordering can invert with distance
and with which side of threshold the noise sits on, so the grid covers both regimes.
Timing is measured once at the reference point; the grid is accuracy + paired stats.
"""
import json
import pathlib
import sys

sys.path.insert(0, "src")
from ketqat_benchmarks.decoder_comparison import run_comparison

GRID = [
    # (distance, rounds, noise, seed) -- 0.008 below ~1% threshold, 0.02 above it.
    (3, 3, 0.008, 7), (3, 3, 0.008, 11),
    (3, 3, 0.02, 7),  (3, 3, 0.02, 11),
    (5, 5, 0.008, 7), (5, 5, 0.008, 11),
    (5, 5, 0.02, 7),  (5, 5, 0.02, 11),
]
SHOTS = 5000

out_dir = pathlib.Path("results/grid")
out_dir.mkdir(parents=True, exist_ok=True)
summary = []
for d, r, p, s in GRID:
    report = run_comparison(d, r, p, SHOTS, seed=s, with_timing=False)
    name = f"d{d}-r{r}-p{str(p).replace('.','')}-s{s}.json"
    (out_dir / name).write_text(json.dumps(report, indent=2, default=str) + "\n")
    row = {"point": name, "publishable": report["publishable"]}
    for e in report["decoders"]:
        if e["available"]:
            row[e["decoder"]] = round(e["unconditional_risk"], 5)
    row["significant_pairs"] = sum(1 for q in report["paired_comparisons"] if q["significant_after_bonferroni"])
    summary.append(row)
    print(f"  {name}: " + " ".join(f"{k}={v}" for k, v in row.items() if k != "point"))

(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"\n{len(GRID)} points written to results/grid/")
