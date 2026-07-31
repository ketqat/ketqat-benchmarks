# ketqat-benchmarks

Benchmark suites, decoder comparisons, fixtures and importers for KetQat.

Separate from [`ketqat-sdk`](https://github.com/ketqat/ketqat-sdk) because the SDK is the
public contract layer and its only runtime dependency is `zod`. Running a decoder
comparison needs Stim, Sinter, PyMatching, BeliefMatching and Tesseract — a dependency set
that must never enter the package other people install.

## What an install contains

The wheel carries the suites, the published results and the citation file, so an installed
copy is the citable thing rather than a pointer at this repository:

```python
from ketqat_benchmarks import data

data.list_suites()        # the six versioned suite declarations
data.list_results()       # every published comparison, including the superseded one
data.load_result("d3-r3-p02-v2.json")
data.citation_path()      # CITATION.cff, as installed
```

Until [#6](https://github.com/ketqat/ketqat-benchmarks/issues/6) the wheel held only the
harness: no suites, no results, no citation, no gate. It went unnoticed because CI installed
`-e .` from the source tree, where every path resolves whether or not it is packaged. The
tests now build a wheel and read *that*, and a `clean-room` CI job installs only the built
wheel and runs the comparison with this checkout nowhere on the path.

## Decoder comparison

```bash
python -m pip install -e ".[decoders]"
python -m ketqat_benchmarks.decoder_comparison --distance 3 --rounds 3 --max-shots 20000
ketqat-benchmarks-gate out.json      # refuses a comparison that is not what it looks like
```

Every decoder sees **identical samples**, proved rather than asserted: one detector/
observable sample is drawn directly from Stim (not Sinter), its bytes are SHA-256 hashed,
and every decoder adapter recomputes that hash from the rows it actually iterated — a
mismatch excludes the decoder from the paired comparison. Two decoders sampled separately
are two experiments; equal shot counts alone prove nothing.

Separation is decided by paired per-shot inference (exact McNemar with Bonferroni across
pairs, paired risk-difference CIs), not by overlap of marginal intervals.

Results carry `is_demo: false` only when they came from a real run. Nothing here fabricates
a number when a dependency is missing — a missing decoder is recorded as not run.
