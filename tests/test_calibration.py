"""Distributional targets from TRUTH.md.

These are design targets for a synthetic teaching dataset, not estimates from any
named survey. They exist so that a future edit cannot silently move a prevalence
into a range that would teach the wrong thing.
"""
import pytest

from conftest import build


@pytest.fixture(scope="module")
def ph():
    return build("public_health", n=30_000, seed=4)


def test_phq9_uses_the_full_scale(ph):
    """The old construction could not reach either end: the floor sat at 2."""
    s = ph.phq9_score
    assert s.min() == 0, f"nobody scores 0 (min={s.min()}): the floor is wrong"
    assert s.max() >= 20, f"nobody reaches the severe range (max={s.max()})"
    assert s.max() <= 27, "PHQ-9 is bounded at 27"


def test_phq9_prevalence_bands(ph):
    s = ph.phq9_score
    minimal = (s < 5).mean()
    moderate = (s >= 10).mean()
    severe = (s >= 20).mean()
    assert 0.45 <= minimal <= 0.65, f"minimal band {minimal:.3f} outside 0.45-0.65"
    assert 0.12 <= moderate <= 0.18, f"moderate {moderate:.3f} outside 0.12-0.18"
    assert 0.008 <= severe <= 0.025, f"severe {severe:.3f} outside 0.008-0.025"


def test_depression_severe_carries_information(ph):
    """It was identically zero in every draw before the generator was rebuilt."""
    assert ph.depression_severe.nunique() == 2


def test_phq9_gradients_run_the_right_way(ph):
    by_q = ph.groupby("wealth_quintile").depression_moderate.mean()
    assert by_q.loc[1] > by_q.loc[5], "poorer quintiles should screen higher, not lower"
    by_sex = ph.groupby("female").depression_severe.mean()
    assert by_sex.loc[1] > by_sex.loc[0], "expected a higher rate among women"


def test_girls_school_distance_splits_urban_and_rural():
    d = build("girls_education", n=20_000, seed=3)
    rural = d[d.urban == 0].distance_to_school_km.mean()
    urban = d[d.urban == 1].distance_to_school_km.mean()
    assert 3.5 <= rural <= 4.5, f"rural mean {rural:.2f} km outside 3.5-4.5"
    assert 1.2 <= urban <= 1.8, f"urban mean {urban:.2f} km outside 1.2-1.8"
    assert rural > urban


@pytest.mark.parametrize("col,rate", [("owns_radio", 0.55), ("owns_mobile", 0.70),
                                      ("owns_bicycle", 0.35)])
def test_targeting_assets_match_their_stated_rates(col, rate):
    """All three were constants before the size-less draws were fixed."""
    d = build("targeting", n=20_000, seed=3)
    got = d[col].mean()
    assert abs(got - rate) < 0.05, f"{col} mean {got:.3f} vs stated {rate}"


def test_rct_takeup_is_partial_in_every_arm():
    d = build("rct_experiment", n=20_000, seed=3)
    for arm, v in d.groupby("treatment_arm").actually_treated.mean().items():
        if arm == "control":
            assert v == 0, "control arm must have zero take-up by definition"
        else:
            assert 0.60 <= v <= 0.90, f"{arm} take-up {v:.3f} outside 0.60-0.90"
