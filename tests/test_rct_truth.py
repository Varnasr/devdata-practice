"""The RCT generator must give back the parameters TRUTH.md says it uses.

If these fail, either the generator changed or TRUTH.md is now lying to learners.
Fix whichever is wrong; do not loosen the tolerance to make the failure go away.
"""
import numpy as np
import pytest

from conftest import build

SEEDS = (1, 42, 123, 777, 2026)
N = 40_000
TOL = 0.01  # log points

# log(multiplier) + mean heterogeneity loading among compliers; see TRUTH.md.
HET = np.log(1 + 0.04 * 0.52 + 0.06 * 0.50)
TRUE_LATE = {
    "cash_transfer": np.log(1.15) + HET,
    "cash_plus_training": np.log(1.22) + HET,
    "training_only": np.log(1.08) + HET,
}
TRUE_SPILLOVER = np.log(1.03)


def _panel(seed):
    df = build("rct_experiment", n=N, seed=seed)
    d = df.dropna(subset=["endline_consumption_usd", "baseline_consumption_usd"]).copy()
    d["lr"] = np.log(d.endline_consumption_usd) - np.log(d.baseline_consumption_usd)
    return d


def _pure_control_mean(d):
    return d[(d.treatment_arm == "control") & (d.spillover_risk == 0)].lr.mean()


@pytest.mark.parametrize("arm", sorted(TRUE_LATE))
def test_wald_late_recovers_true_complier_effect(arm):
    got = []
    for seed in SEEDS:
        d = _panel(seed)
        a = d[d.treatment_arm == arm]
        got.append((a.lr.mean() - _pure_control_mean(d)) / a.actually_treated.mean())
    mean = float(np.mean(got))
    assert abs(mean - TRUE_LATE[arm]) < TOL, (
        f"{arm}: Wald LATE {mean:+.4f} vs TRUTH.md {TRUE_LATE[arm]:+.4f} "
        f"(per-seed: {[round(x, 4) for x in got]})"
    )


def test_spillover_is_recoverable_from_pure_control_villages():
    got = []
    for seed in SEEDS:
        d = _panel(seed)
        exposed = d[(d.treatment_arm == "control") & (d.spillover_risk == 1)].lr.mean()
        got.append(exposed - _pure_control_mean(d))
    mean = float(np.mean(got))
    assert abs(mean - TRUE_SPILLOVER) < TOL, (
        f"spillover {mean:+.4f} vs TRUTH.md {TRUE_SPILLOVER:+.4f}"
    )


def test_contaminated_control_group_biases_itt_toward_zero():
    """The lesson the design exists to teach, asserted so it cannot be lost."""
    d = _panel(123)
    pure = _pure_control_mean(d)
    allc = d[d.treatment_arm == "control"].lr.mean()
    assert allc > pure, "controls in treatment villages should do better than pure controls"
    for arm in TRUE_LATE:
        m = d[d.treatment_arm == arm].lr.mean()
        assert (m - allc) < (m - pure), f"{arm}: contaminated ITT should be smaller"


def test_spillover_risk_is_not_an_alias_of_the_control_dummy():
    """The original defect, pinned directly."""
    d = build("rct_experiment", n=20_000, seed=7)
    ctrl = d[d.treatment_arm == "control"]
    assert ctrl.spillover_risk.nunique() == 2, (
        "spillover_risk must vary among controls: some sit in pure-control villages"
    )
    assert (d[d.treatment_arm != "control"].spillover_risk == 0).all(), (
        "treated units are not spillover-exposed by this definition"
    )


def test_village_structure_has_pure_control_villages():
    d = build("rct_experiment", n=20_000, seed=7)
    share = d.groupby("village_id").village_treated_share.first()
    assert (share == 0).any(), "no pure-control villages: spillover is unidentified"
    assert (share > 0).any(), "no treatment villages"
