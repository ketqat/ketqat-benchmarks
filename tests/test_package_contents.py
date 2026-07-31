"""What the wheel actually contains (#6).

The defect these tests exist for: the wheel shipped `decoder_comparison.py` and nothing
else -- no suites, no results, no citation file, no gate. Nobody noticed because CI
installs `-e .` from the source tree, where every path resolves whether it is packaged or
not.

So these tests **build the wheel and read it**, rather than asking the importable package
what it can see. Under an editable install the two questions have different answers, and
only the first one is about what a user gets.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A freshly built wheel. Built once; every test below reads this archive."""
    output = tmp_path_factory.mktemp("wheel")
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(output), str(ROOT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot build a wheel here: {result.stderr.strip()[-300:]}")

    wheels = list(output.glob("*.whl"))
    assert len(wheels) == 1, f"expected one wheel, got {wheels}"
    return wheels[0]


@pytest.fixture(scope="module")
def wheel_names(wheel: Path) -> list[str]:
    """Every path inside the built wheel."""
    with zipfile.ZipFile(wheel) as archive:
        return archive.namelist()


def test_the_wheel_carries_every_suite(wheel_names: list[str]) -> None:
    # Six, not four: the SDK carries six versioned suite declarations and all six are
    # published, with the discrepancy recorded in suites/PROVENANCE.md rather than
    # silently resolved.
    packaged = {name for name in wheel_names if name.startswith("ketqat_benchmarks/suites/")}
    source = {path.name for path in (ROOT / "suites").glob("*.yaml")}
    assert source, "the repository has no suites; this test would pass vacuously"
    for suite in source:
        assert f"ketqat_benchmarks/suites/{suite}" in packaged, f"{suite} is not in the wheel"
    # The provenance file is what ties a published suite back to its origin commit.
    assert "ketqat_benchmarks/suites/PROVENANCE.md" in packaged


def test_the_wheel_carries_the_results_including_the_superseded_marker(
    wheel_names: list[str],
) -> None:
    packaged = {name for name in wheel_names if name.startswith("ketqat_benchmarks/results/")}
    source = {
        str(path.relative_to(ROOT / "results")) for path in (ROOT / "results").rglob("*.json")
    }
    assert source, "the repository has no results; this test would pass vacuously"
    for result in source:
        assert f"ketqat_benchmarks/results/{result}" in packaged, f"{result} is not in the wheel"
    # A superseded result stays, with its marker. Shipping the replacement without the
    # marker is how a superseded number gets quoted as current.
    assert "ketqat_benchmarks/results/SUPERSEDED.md" in packaged


def test_the_wheel_can_be_cited(wheel_names: list[str]) -> None:
    # CITATION.cff was in the sdist and not in the wheel, and `pip install` resolves the
    # wheel -- so the installed package could not be cited.
    assert "ketqat_benchmarks/CITATION.cff" in wheel_names


def test_the_wheel_carries_the_gate(wheel_names: list[str]) -> None:
    assert "ketqat_benchmarks/gate.py" in wheel_names
    assert "ketqat_benchmarks/data.py" in wheel_names


def test_the_gate_is_installed_as_a_command(wheel: Path, wheel_names: list[str]) -> None:
    """Read the entry point out of the wheel, not out of pyproject.toml.

    pyproject.toml is the file that was wrong; asking it whether the gate is installed
    would be asking the defect to report itself.
    """
    declarations = [name for name in wheel_names if name.endswith("dist-info/entry_points.txt")]
    assert declarations, "the wheel declares no console scripts"
    with zipfile.ZipFile(wheel) as archive:
        text = archive.read(declarations[0]).decode("utf-8")
    assert "ketqat-benchmarks-gate" in text
    assert "ketqat_benchmarks.gate:main" in text


def _run_against_the_wheel(wheel: Path, tmp_path: Path, body: str) -> str:
    """Run `body` with only the wheel's contents importable.

    Not with the source tree on the path. Under an editable install the accessors
    resolve `suites/` and `results/` from the repository root, where they live for
    humans to find -- so an accessor test run in this checkout would pass whether or not
    the data was packaged, which is the exact blindness that let #6 ship.
    """
    extracted = tmp_path / "site"
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)
    result = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PYTHONPATH": str(extracted), "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr.strip()[-2000:]
    return result.stdout


def test_the_packaged_data_is_reachable_by_the_accessors(wheel: Path, tmp_path: Path) -> None:
    output = _run_against_the_wheel(
        wheel,
        tmp_path,
        """
from ketqat_benchmarks import data

suites = data.list_suites()
assert len(suites) >= 6, f"expected every published suite, found {suites}"
assert data.suite_path(suites[0]).is_file()

results = data.list_results()
assert "d3-r3-p02-v2.json" in results, results
assert data.load_result("d3-r3-p02-v2.json")["decoders_run"] == 3
assert data.citation_path().is_file()

# A typo and a package built without its data are otherwise the same error, so the
# message has to list what is available.
try:
    data.suite_path("no-such-suite")
    raise AssertionError("a missing suite must raise")
except FileNotFoundError as error:
    assert "Available:" in str(error), error

print(len(suites), len(results))
""",
    )
    suite_count, result_count = output.split()
    assert int(suite_count) >= 6 and int(result_count) >= 9


def test_the_gate_still_refuses_a_bad_report(wheel: Path, tmp_path: Path) -> None:
    """Moving the gate into the package must not have softened it.

    Each mutation is a way a comparison looks fine and is not. Run against the wheel,
    because the gate a user gets is the one in the wheel.
    """
    _run_against_the_wheel(
        wheel,
        tmp_path,
        """
import json
from ketqat_benchmarks.gate import check
from ketqat_benchmarks.data import load_result

report = load_result("d3-r3-p02-v2.json")
assert check(report) == [], check(report)

def mutate(**changes):
    copy = json.loads(json.dumps(report))
    for path, value in changes.items():
        copy[path] = value
    return copy

dropped = mutate(decoders=report["decoders"][:2])
assert any("of 3 decoders ran" in f for f in check(dropped)), check(dropped)

unshared = json.loads(json.dumps(report))
unshared["decoders"][1]["consumed_sample_sha256"] = "0" * 64
assert any("did not provably consume" in f for f in check(unshared)), check(unshared)

claimed = json.loads(json.dumps(report))
claimed["experiment"]["sampling"] = "Sinter task collection"
assert any("claims Sinter" in f for f in check(claimed)), check(claimed)

assert any("is_demo" in f for f in check(mutate(is_demo=True)))
assert any("not publishable" in f for f in check(mutate(publishable=False)))
assert any("no paired comparisons" in f for f in check(mutate(paired_comparisons=[])))
""",
    )
