"""Shared fixtures. Tests import the package from the repository root."""
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from generate import GENERATORS  # noqa: E402

# Small enough that the whole suite runs in CI in well under a minute, large
# enough that a prevalence assertion is not dominated by sampling noise.
SMOKE_N = 2000


@lru_cache(maxsize=None)
def _build_cached(name, n, seed):
    import importlib
    info = GENERATORS[name]
    kwargs = {"seed": seed}
    if info["size_param"]:
        kwargs[info["size_param"]] = n
    return importlib.import_module(info["module"]).generate(**kwargs)


def build(name, n=SMOKE_N, seed=11):
    """Generate one dataset by registry name.

    Cached: the suite asks for the same (name, n, seed) from several modules, and
    regenerating 40,000 rows each time made the run take minutes. Callers must not
    mutate what they get back: copy first if you need to.
    """
    return _build_cached(name, n, seed)


@pytest.fixture(scope="session")
def generator_names():
    return sorted(GENERATORS)
