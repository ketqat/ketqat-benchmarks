#!/usr/bin/env python3
"""CI gate: three decoders, provably on one sample, with derived publishability.

A comparison that quietly drops to one decoder, or whose entrants did not provably
consume the shared sample, still emits a valid-looking report -- this fails the build
instead. Nothing here asserts is_demo; it checks what the library derived.
"""
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
ran = [d for d in report["decoders"] if d["available"]]
missing = [(d["decoder"], d["not_run_reason"]) for d in report["decoders"] if not d["available"]]

failures = []
if len(ran) < 3:
    failures.append(f"{len(ran)} of 3 decoders ran; missing: {missing}")
master = report["experiment"]["sample_sha256"]
for d in ran:
    if d["consumed_sample_sha256"] != master:
        failures.append(f"{d['decoder']} did not provably consume the shared sample")
if report.get("is_demo") is not False:
    failures.append("report derived is_demo != False (nothing real ran?)")
if report.get("publishable") is not True:
    failures.append("report is not publishable by its own derivation")
if not report.get("paired_comparisons"):
    failures.append("no paired comparisons present")
if "Sinter" in report["experiment"]["sampling"] and "not Sinter" not in report["experiment"]["sampling"]:
    failures.append("sampling claims Sinter; this harness samples directly from Stim")

if failures:
    print("FAIL:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS: {len(ran)} decoders, sample {master[:12]}, publishable derived True, "
      f"{len(report['paired_comparisons'])} paired comparisons")
