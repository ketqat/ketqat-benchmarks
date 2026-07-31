# Superseded results

## d3-r3-p02.json (schema 0.1, commit 8545166)

Retained for the record; **do not consume or cite as authoritative.** Superseded by
`d3-r3-p02-v2.json` (schema 0.2). Defects in the 0.1 result:

- The module docstring claimed a "single Sinter task collection"; sampling was in fact
  direct shared-Stim. The claim was false; the v2 module states the method truthfully.
- Shared samples were "proved" by equal shot counts, which proves nothing. v2 hashes the
  sample and every decoder recomputes the hash from the arrays it iterated.
- Separation was judged by Wilson-interval overlap (1 of 3 pairs). Paired McNemar on the
  same shots separates **all 3 pairs** decisively — overlap of marginals was the wrong test.
- `tracemalloc` was reported as memory; it is Python-heap allocation only. v2 reports
  child-process peak RSS and says so.
- Single-shot latency and batch throughput were conflated. v2 separates them; PyMatching's
  batch API is ~16× its single-shot loop.
- `is_demo: false` was hardcoded; v2 derives `is_demo` and `publishable`.
