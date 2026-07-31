#!/usr/bin/env python3
"""CI gate: the leaderboard must not silently shrink to one entrant.

A comparison that quietly drops to a single decoder still produces a valid-looking
report, which is exactly the failure this guards -- so a missing decoder fails the
build rather than shortening a table.
"""
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
ran = [d for d in report["decoders"] if d["available"]]
missing = [(d["decoder"], d["not_run_reason"]) for d in report["decoders"] if not d["available"]]

if len(ran) < 3:
    print(f"FAIL: {len(ran)} of 3 decoders ran. Missing: {missing}")
    sys.exit(1)
if report["is_demo"] is not False:
    print("FAIL: results are marked demo")
    sys.exit(1)
if len({d["shots"] for d in ran}) != 1:
    print("FAIL: decoders saw different sample counts, so this is not one comparison")
    sys.exit(1)
print(f"PASS: {len(ran)} decoders, {ran[0]['shots']} identical shots each, is_demo=False")
