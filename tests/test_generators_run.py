"""Every registered generator must run, and the registry must match the disk.

CI previously ran only `generate.py --list`, which imports no generator at all.
Thirty-six modules were therefore never executed by any automated check.
"""
import pathlib

import pytest

from conftest import build
from generate import GENERATORS


def test_registry_matches_disk():
    """A module on disk but not in GENERATORS is invisible; the reverse crashes."""
    disk = {p.stem for p in (pathlib.Path(__file__).parent.parent / "generators").glob("*.py")}
    disk -= {"__init__", "utils"}
    assert disk == set(GENERATORS), (
        f"on disk but unregistered: {sorted(disk - set(GENERATORS))}; "
        f"registered but missing: {sorted(set(GENERATORS) - disk)}"
    )


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_generator_runs_and_is_non_trivial(name):
    df = build(name)
    assert len(df) > 0, f"{name} produced no rows"
    assert df.shape[1] >= 5, f"{name} produced only {df.shape[1]} columns"
    assert not df.columns.duplicated().any(), f"{name} has duplicate column names"


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_same_seed_same_data(name):
    """Reproducibility is the whole point of shipping a seed."""
    assert build(name, seed=5).equals(build(name, seed=5)), f"{name} is not reproducible"


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_different_seed_different_data(name):
    """A seed that changes nothing means the seed is not wired through."""
    assert not build(name, seed=5).equals(build(name, seed=6)), (
        f"{name} ignores its seed argument"
    )
