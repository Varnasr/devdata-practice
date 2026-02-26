"""
Generator 2: RCT Experiment Data
─────────────────────────────────
Simulates a multi-arm randomized controlled trial in a development context
(e.g., cash transfer, school feeding, deworming).

Realistic features:
  • Stratified randomization by district and gender
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

    # --- Treatment assignment (stratified by district × gender) ---
    arms = ["control", "cash_transfer", "cash_plus_training", "training_only"]
    treatment = np.empty(n, dtype="<U20")

    # Stratify assignment
    strata = [f"{d}_{g}" for d, g in zip(districts, female)]
    unique_strata = list(set(strata))
    for s in unique_strata:
        mask = np.array([x == s for x in strata])
        n_s = mask.sum()
        perm = rng.permutation(n_s)
        arm_assign = np.array([arms[i % len(arms)] for i in perm])
        treatment[mask] = arm_assign

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

    # --- Endline outcome ---
    time_trend = 1.03  # 3% general improvement
    noise = np.exp(rng.normal(0, 0.15, n))
    endline_consumption = np.round(
        baseline_consumption * time_trend * te_multiplier * (1 + het_female + het_poor) * noise, 2
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

    # --- Spillover flag (10% of control units in treated villages) ---
    spillover_risk = np.zeros(n, dtype=int)
    for d in np.unique(districts):
        d_mask = districts == d
        has_treated = (treatment[d_mask] != "control").any()
        if has_treated:
            ctrl = d_mask & (treatment == "control")
            spillover_risk[ctrl] = 1

    # Build DataFrame
    df = pd.DataFrame({
        "participant_id": ids,
        "district": districts,
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
        "spillover_risk": spillover_risk,
    })

    # Additional missingness in baseline vars (pre-existing survey issues)
    df = inject_missing(df,
        columns=["baseline_consumption_usd", "education_years", "household_size"],
        rates=[0.03, 0.02, 0.01],
        rng=rng, mechanism="MCAR")

    return df
