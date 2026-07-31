"""Reaching the suites and results from an installed package (#6).

The wheel shipped `decoder_comparison.py` and nothing else: no suites, no results, no
citation file. Publishing suites so they can be cited at a version -- without citing an
SDK commit -- is the reason this repository exists, and a wheel without them does not
deliver it.

Nothing noticed because CI installs `-e .` from the source tree, where every path
resolves whether it is packaged or not. That is why the test for this reads the **built
wheel**.

Data files live under the package directory rather than at the top level of
`site-packages`, so installing this does not put `suites/` and `results/` next to
everybody else's modules.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any


def _directory(name: str) -> Path:
    """A packaged data directory as a real path.

    `as_file` would be the general form, but these are plain files in a wheel with no
    zip import in play, and returning a `Path` keeps callers simple.
    """
    return Path(str(resources.files("ketqat_benchmarks") / name))


def suites_directory() -> Path:
    return _directory("suites")


def results_directory() -> Path:
    return _directory("results")


def list_suites() -> list[str]:
    """Suite file stems, sorted. Empty means the package was built without its data."""
    directory = suites_directory()
    if not directory.is_dir():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


def suite_path(name: str) -> Path:
    """The path of one suite declaration.

    Raises:
        FileNotFoundError: naming what is available, because a typo and a package built
            without its data are otherwise the same error.
    """
    path = suites_directory() / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(
            f"No packaged suite named {name!r}. Available: {', '.join(list_suites()) or 'none'}."
        )
    return path


def list_results() -> list[str]:
    """Result file names relative to the results directory, sorted, including the grid."""
    directory = results_directory()
    if not directory.is_dir():
        return []
    return sorted(
        str(path.relative_to(directory)) for path in directory.rglob("*.json")
    )


def load_result(name: str) -> dict[str, Any]:
    """One published comparison, by name relative to the results directory."""
    path = results_directory() / name
    if not path.is_file():
        raise FileNotFoundError(
            f"No packaged result named {name!r}. Available: {', '.join(list_results()) or 'none'}."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def citation_path() -> Path:
    """`CITATION.cff` as installed.

    It was in the sdist and not in the wheel, and `pip install` resolves the wheel -- so
    the installed package could not be cited.
    """
    path = _directory("CITATION.cff")
    if not path.is_file():
        raise FileNotFoundError(
            "CITATION.cff is not in this install. The package was built without it; see "
            "ketqat-benchmarks#6."
        )
    return path
