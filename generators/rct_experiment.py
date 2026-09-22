"""
Generator 2: RCT Experiment Data
─────────────────────────────────
Simulates a multi-arm randomized controlled trial in a development context
(e.g., cash transfer, school feeding, deworming).

Realistic features:
  • Two-level design: villages are randomised into treatment or pure-control,
    then individuals within treatment villages are stratified-randomised by arm
  • Stratified randomization by village and gender
  • Baseline and endline observations
  • Partial compliance (take-up < 100%)
  • Attrition correlated with treatment arm and baseline characteristics
  • Spillover potential (flagged)
  • Lee bounds-compatible structure
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_individuals: int = 25000, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    # --- Baseline characteristics ---
    ids = [f"P-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.30)
    female = rng.binomial(1, 0.52, n)
    age = rng.integers(18, 65, n)
    educ_years = np.clip(rng.normal(7, 3.5, n), 0, 18).astype(int)
    hh_size = rng.choice(range(1, 10), n, p=[0.03, 0.06, 0.12, 0.20, 0.22, 0.18, 0.10, 0.06, 0.03])

    # Baseline outcome (e.g., monthly consumption per capita, USD PPP)
    baseline_score = (
        3.5 + 0.15 * educ_years + 0.3 * urban.astype(float)
        - 0.05 * hh_size + rng.normal(0, 0.5, n)
    )
    baseline_consumption = np.round(np.exp(baseline_score), 2)

    # Baseline secondary outcome (e.g., food security 0-27 HFIAS)
    baseline_food_insecurity = np.clip(
        rng.poisson(12 - 0.4 * educ_years - 2 * urban.astype(float) + 0.3 * hh_size),
        0, 27
    )

    # --- Villages ---
    # Individuals are clustered in villages within districts. This level exists so
    # spillover exposure can vary: without it every district contained treated units
    # by construction, and the spillover flag below was an exact alias of the control
    # dummy (see tests/test_no_degenerate_columns.py, which now pins that down).
    villages_per_district = 8
    village_idx = rng.integers(0, villages_per_district, n)
    village_id = np.array([f"{d}-V{v}" for d, v in zip(districts, village_idx)])

    # --- Treatment assignment (two-level) ---
    # 70% of villages are treatment villages; the remaining 30% are pure controls,
    # where nobody is offered anything. Within a treatment village, individuals are
    # stratified-randomised across the four arms by gender.
    arms = ["control", "cash_transfer", "cash_plus_training", "training_only"]
    treatment = np.full(n, "control", dtype="<U20")

    all_villages = np.unique(village_id)
    is_treatment_village = dict(
        zip(all_villages, rng.binomial(1, 0.70, len(all_villages)).astype(bool))
    )
    village_is_treated = np.array([is_treatment_village[v] for v in village_id])

    strata = np.array([f"{v}_{g}" for v, g in zip(village_id, female)])
    for st in np.unique(strata[village_is_treated]):
        mask = (strata == st) & village_is_treated
        n_s = int(mask.sum())
        perm = rng.permutation(n_s)
        treatment[mask] = np.array([arms[i % len(arms)] for i in perm])

    # Control units living in a treatment village are exposed to spillovers;
    # control units in pure-control villages are not.
    exposed_control = (treatment == "control") & village_is_treated

    # --- Compliance (take-up) ---
    # Control: 0% take-up (by definition)
    # Treatment arms: 65-85% take-up
    actually_treated = np.zeros(n, dtype=int)
    for arm in arms[1:]:
        mask = treatment == arm
        compliance_rate = rng.uniform(0.65, 0.85)
        actually_treated[mask] = rng.binomial(1, compliance_rate, mask.sum())

    # --- True treatment effects (heterogeneous) ---
    # Cash transfer: +15% consumption, cash+training: +22%, training: +8%
    te_multiplier = np.ones(n)
    te_multiplier[actually_treated.astype(bool) & (treatment == "cash_transfer")] = 1.15
    te_multiplier[actually_treated.astype(bool) & (treatment == "cash_plus_training")] = 1.22
    te_multiplier[actually_treated.astype(bool) & (treatment == "training_only")] = 1.08

    # Heterogeneity: larger effect for women, poorer baseline
    het_female = 0.04 * female * actually_treated
    het_poor = 0.06 * (baseline_consumption < np.median(baseline_consumption)).astype(float) * actually_treated

    # --- Spillover onto untreated neighbours ---
    # Cash landing in a village lifts local demand, so untreated households in a
    # treatment village gain a little even though they were offered nothing. The
    # true spillover is +3% on consumption; see TRUTH.md. It is why an ITT computed
    # against *all* controls is biased toward zero, and why the clean comparison is
    # against pure-control villages only.
    spillover_multiplier = np.where(exposed_control, 1.03, 1.0)

    # --- Endline outcome ---
    time_trend = 1.03  # 3% general improvement
    noise = np.exp(rng.normal(0, 0.15, n))
    endline_consumption = np.round(
        baseline_consumption * time_trend * te_multiplier * spillover_multiplier
        * (1 + het_female + het_poor) * noise, 2
    )

    # Endline food insecurity (should improve with treatment)
    fi_effect = np.zeros(n)
    fi_effect[actually_treated.astype(bool) & (treatment == "cash_transfer")] = -3
    fi_effect[actually_treated.astype(bool) & (treatment == "cash_plus_training")] = -5
    fi_effect[actually_treated.astype(bool) & (treatment == "training_only")] = -2
    endline_food_insecurity = np.clip(
        baseline_food_insecurity + fi_effect + rng.normal(0, 2, n), 0, 27
    ).astype(int)

    # --- Attrition (correlated with arm & baseline) ---
    attrition_prob = 0.08 + 0.03 * (treatment == "control").astype(float)
    attrition_prob += 0.02 * (1 - urban.astype(float))
    attrition_prob -= 0.01 * (educ_years / 18)
    attrited = rng.binomial(1, np.clip(attrition_prob, 0.02, 0.25), n).astype(bool)

    # --- Spillover exposure ---
    # Share of each village actually offered treatment, and the exposure flag for
    # control units sitting inside a treatment village. Controls in pure-control
    # villages score 0 and are the clean comparison group; controls in treatment
    # villages are the ones a spillover analysis has to worry about.
    assigned = (treatment != "control").astype(float)
    vs = pd.Series(assigned).groupby(pd.Series(village_id)).transform("mean")
    village_treated_share = np.round(vs.to_numpy(), 4)
    spillover_risk = exposed_control.astype(int)

    # Build DataFrame
    df = pd.DataFrame({
        "participant_id": ids,
        "district": districts,
        "village_id": village_id,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "education_years": educ_years,
        "household_size": hh_size,
        "treatment_arm": treatment,
        "actually_treated": actually_treated,
        "baseline_consumption_usd": baseline_consumption,
        "baseline_food_insecurity": baseline_food_insecurity,
        "endline_consumption_usd": np.where(attrited, np.nan, endline_consumption),
        "endline_food_insecurity": np.where(attrited, np.nan, endline_food_insecurity).astype(float),
        "attrited": attrited.astype(int),
        "village_treated_share": village_treated_share,
        "spillover_risk": spillover_risk,
    })

    # Additional missingness in baseline vars (pre-existing survey issues)
    df = inject_missing(df,
        columns=["baseline_consumption_usd", "education_years", "household_size"],
        rates=[0.03, 0.02, 0.01],
        rng=rng, mechanism="MCAR")

    return df
