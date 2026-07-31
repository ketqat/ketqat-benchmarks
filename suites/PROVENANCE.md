# Suite provenance

Published from `ketqat-sdk` @ `4ca04fa9cc9da75265b61d68d02772135086638d` (no history rewrite; SDK copies remain as
shims until clean installs pass without them).

The Product 3.0 brief said "the four existing versioned SDK suites"; the SDK carries
**six** versioned suite declarations. All six are published rather than guessing which
four were meant — the discrepancy is recorded here instead of silently resolved:

| suite | version | origin |
|---|---|---|
| grover-search-local | 0.1.0 | examples/algorithms/grover-search.yaml |
| phase-estimation-textbook | 0.1.0 | examples/algorithms/phase-estimation.yaml |
| randomized-benchmarking-clifford | 0.1.0 | examples/protocols/randomized-benchmarking.yaml |
| surface-code-memory-decoder-comparison | 0.1.0 | examples/qec/decoder-comparison.yaml |
| surface-code-memory-readout-limited | 0.1.0 | examples/qec/readout-limited-memory.yaml |
| surface-code-memory-mwpm | 0.1.0 | examples/qec/surface-code-memory.yaml |
