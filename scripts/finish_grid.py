#!/usr/bin/env python3
"""Finish the two d=5 above-threshold points at 2000 shots.

2000 rather than 5000: above threshold at d=5 the error rate is high, so discordant
counts are plentiful and the paired test does not need the larger sample -- and
Tesseract at d=5 costs ~220ms/shot, which makes 5000 a runtime decision, recorded here
rather than hidden.
"""
import json, pathlib, sys
sys.path.insert(0, "src")
from ketqat_benchmarks.decoder_comparison import run_comparison

out = pathlib.Path("results/grid")
for seed in (7, 11):
    report = run_comparison(5, 5, 0.02, 2000, seed=seed, with_timing=False)
    name = f"d5-r5-p002-s{seed}.json"
    (out / name).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"done {name} publishable={report['publishable']}")
print("grid complete")
