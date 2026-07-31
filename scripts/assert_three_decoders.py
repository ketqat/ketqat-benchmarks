#!/usr/bin/env python3
"""Shim. The gate now lives in the package: `ketqat_benchmarks.gate`.

It was here, outside `src/`, and therefore absent from the wheel -- so an install could
produce a comparison and not check it (#6). This file stays so CI steps and existing
instructions keep working, and so the two can never disagree: there is one
implementation.
"""
import sys

from ketqat_benchmarks.gate import main

if __name__ == "__main__":
    sys.exit(main())
