"""The guard for the bug that made this repository's data quietly useless.

`rng.binomial(1, 0.55)` without a size argument returns a *scalar*, which numpy
then broadcasts across the whole column. Nothing errors, the file writes, the row
count is right, and the variable is a constant. It reached the repository in two
shapes:

    owns_radio = rng.binomial(1, 0.55)                       # every household: 1
    barrier_marriage = np.where(cond, rng.binomial(1, .25), 0)  # every girl: 0

Nineteen call sites across nine generators were affected, producing fifteen
constant columns. Four of them were assets in `targeting`, which exists to teach
proxy means testing, and a PMT whose asset predictors do not vary is not a PMT.

The tell that confirmed the diagnosis is still in `girls_education`: `barrier_cost`
on the neighbouring line takes an *array* probability, so numpy returned a vector
and that column was always fine.

This test also catches the second shape of the same failure: a column that varies
but is a perfect alias of another column.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from conftest import build
from generate import GENERATORS

# Legitimately constant, with the reason.
EXPECTED_CONSTANT = {
    # Item parameters are fixed by construction in a wide-format IRT table: every
    # respondent answers the same 30 items, so difficulty and discrimination do
    # not vary by row. They are shipped as columns so the frame is self-describing.
    ("irt_assessment", "item_{}_difficulty"),
    ("irt_assessment", "item_{}_discrimination"),
    # A poverty line is a single threshold applied to everyone. That is what it is.
    ("targeting", "poverty_line_usd"),
}
_CONST_EXACT = {(g, c) for g, c in EXPECTED_CONSTANT if "{}" not in c}
_CONST_ITEM = {(g, c) for g, c in EXPECTED_CONSTANT if "{}" in c}


def _is_expected_constant(gen, col):
    if (gen, col) in _CONST_EXACT:
        return True
    return any(
        g == gen and col == c.format(i)
        for g, c in _CONST_ITEM
        for i in range(1, 51)
    )


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_no_unexpected_constant_columns(name):
    df = build(name, n=3000)
    offenders = [
        c for c in df.columns
        if df[c].nunique(dropna=True) <= 1 and not _is_expected_constant(name, c)
    ]
    assert not offenders, (
        f"{name}: columns with no variation {offenders}. Usually a size-less RNG "
        f"draw. See this module's docstring."
    )


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_no_all_null_columns(name):
    df = build(name, n=3000)
    empty = [c for c in df.columns if df[c].isna().all()]
    assert not empty, f"{name}: columns that are entirely null {empty}"


# Pairs where one column is defined as the other, by design rather than by accident.
EXPECTED_ALIASES = {
    # Categorical eligibility for the female-headed-household window *is* the
    # female-head indicator; the dataset ships both so the targeting rule is legible.
    ("targeting", frozenset({"head_female", "categorical_eligible_fhh"})),
    ("targeting", frozenset({"head_disabled", "categorical_eligible_disabled"})),
}


@pytest.mark.parametrize("name", sorted(GENERATORS))
def test_no_accidental_binary_aliases(name):
    """A binary column that is an exact function of another carries no information.

    `rct_experiment.spillover_risk` was precisely this: 1 for every control and 0
    for every treated unit, because randomisation was stratified within district so
    every district always contained treated units. It is now defined at village
    level, and pure-control villages give it real variation.
    """
    df = build(name, n=3000)
    binary = [c for c in df.columns if df[c].nunique(dropna=True) == 2]
    offenders = []
    for a, b in itertools.combinations(binary, 2):
        if (name, frozenset({a, b})) in EXPECTED_ALIASES:
            continue
        tab = pd.crosstab(df[a].astype(str), df[b].astype(str))
        if tab.shape != (2, 2):
            continue
        v = tab.to_numpy()
        if (np.diag(v) == 0).all() or (np.diag(np.fliplr(v)) == 0).all():
            offenders.append((a, b))
    assert not offenders, (
        f"{name}: perfectly collinear binary pairs {offenders}. One is an alias of "
        f"the other, so a model including both drops a term."
    )
