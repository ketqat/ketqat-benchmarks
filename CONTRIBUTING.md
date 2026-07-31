# Contributing

- Open an issue before substantial work.
- Feature branches and PRs only; CI must pass, including the three-decoder gate.
- A result is publishable only if every entrant provably consumed the shared sample
  (hash match) and at least two entrants ran. Do not weaken that gate.
- New decoders: add an accuracy adapter that hashes the sample as it consumes it, and a
  timing branch in the child worker. A decoder that cannot run is recorded as not run.
