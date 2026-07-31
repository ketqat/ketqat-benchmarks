# ketqat-benchmarks

Benchmark suites, decoder comparisons, fixtures and importers for KetQat.

Separate from [`ketqat-sdk`](https://github.com/ketqat/ketqat-sdk) because the SDK is the
public contract layer and its only runtime dependency is `zod`. Running a decoder
comparison needs Stim, Sinter, PyMatching, BeliefMatching and Tesseract — a dependency set
that must never enter the package other people install.

## Decoder comparison

```bash
python -m pip install -e ".[decoders]"
python -m ketqat_benchmarks.decoder_comparison --distance 3 --rounds 3 --max-shots 20000
```

Every decoder sees **identical samples and identical stopping rules**, because all of them
run through one Sinter task collection rather than three sampling loops. That is what makes
the comparison a comparison; two decoders sampled separately are two experiments.

Results carry `is_demo: false` only when they came from a real run. Nothing here fabricates
a number when a dependency is missing — a missing decoder is recorded as not run.
