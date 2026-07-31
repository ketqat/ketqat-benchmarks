# CLAUDE.md

`ketqat-benchmarks` holds benchmark suites, decoder comparisons, fixtures and importers.
Heavy scientific dependencies (Stim, Sinter, PyMatching, BeliefMatching, Tesseract) live
here so `ketqat-sdk` stays zod-only.

## Non-negotiables

- **Sampling is direct shared-Stim sampling.** Every decoder adapter hashes the sample
  as it consumes it; a mismatched hash excludes the decoder from the paired comparison.
  Do not claim Sinter is used unless it is.
- **Separation is paired inference** (exact McNemar + paired risk-difference CI,
  Bonferroni across pairs). Marginal Wilson intervals are context, never the verdict.
- **Rank uses unconditional risk** (failures + abstentions over all shots), so
  abstention cannot improve position.
- **Latency ≠ throughput.** Single-shot latency (warm-up, randomized repeats) and batch
  throughput (library batch APIs) are separate numbers, measured in a spawned child
  process per decoder. Memory is child `ru_maxrss`; tracemalloc is never reported as memory.
- `is_demo` and `publishable` are **derived**, not asserted. A missing decoder is
  recorded as not run, never a row of zeros.
- Superseded results stay in `results/` with a marker; never overwrite or delete them.

## Commands

```bash
python -m pip install -e ".[decoders]" pytest
python -m pytest tests -q
python -m ketqat_benchmarks.decoder_comparison --distance 3 --rounds 3 --noise 0.02 --max-shots 20000 --output out.json
python scripts/assert_three_decoders.py out.json
```
