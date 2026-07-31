# Security

Report vulnerabilities privately via GitHub security advisories on this repository.

This repository executes no untrusted input: it samples circuits with Stim and runs
published decoder libraries on them. The realistic surface is dependency supply chain
(decoders are compiled extensions) and result integrity — results carry sample, DEM and
circuit hashes plus a reproducibility hash so tampering is detectable.
