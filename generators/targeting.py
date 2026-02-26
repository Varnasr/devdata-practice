"""
Generator 9: Program Targeting (Proxy Means Test)
─────────────────────────────────────────────────
Simulates household-level data for targeting social protection programs using
proxy means testing (PMT). Includes true consumption, predicted PMT score,
eligibility, and inclusion/exclusion errors.

Rows: ~20k households.

Realistic features:
  • PMT formula using observable correlates of consumption
  • Type I (exclusion) and Type II (inclusion) errors
  • Community-based targeting comparison
  • Categorical targeting (e.g., female-headed, elderly)
  • Benefit amount with phase-out
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_households: int = 20000, seed: int = 505) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    # --- Household characteristics ---
    hh_ids = household_ids(rng, n)
    districts, urban = pick_districts(rng, n, urban_share=0.30)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    hh_size = rng.choice(range(1, 11), n,
                         p=[0.03, 0.06, 0.10, 0.16, 0.20, 0.18, 0.13, 0.08, 0.04, 0.02])
    n_children = np.clip(rng.poisson(hh_size * 0.35), 0, hh_size - 1)
    n_elderly = np.clip(rng.poisson(0.3), 0, min(3, max(0, hh_size.max())))
    n_elderly = np.minimum(n_elderly, hh_size - n_children)
    head_female = rng.binomial(1, 0.28, n)
    head_age = rng.integers(20, 75, n)
    head_educ = np.clip(rng.normal(5.5, 3.5, n), 0, 18).astype(int)
    head_employed = rng.binomial(1, 0.65, n)
    head_disabled = rng.binomial(1, 0.05, n)

    # Housing
    wall_permanent = rng.binomial(1, 0.35 + 0.15 * urban.astype(float))
    roof_permanent = rng.binomial(1, 0.40 + 0.15 * urban.astype(float))
    rooms = np.clip(rng.poisson(2), 1, 8)
    has_electricity = rng.binomial(1, 0.30 + 0.30 * urban.astype(float))
    has_piped_water = rng.binomial(1, 0.20 + 0.25 * urban.astype(float))
    has_flush_toilet = rng.binomial(1, 0.15 + 0.20 * urban.astype(float))

    # Assets
    owns_radio = rng.binomial(1, 0.55)
    owns_tv = rng.binomial(1, 0.25 + 0.15 * urban.astype(float))
    owns_mobile = rng.binomial(1, 0.70)
    owns_bicycle = rng.binomial(1, 0.35)
    owns_motorcycle = rng.binomial(1, 0.10 + 0.05 * urban.astype(float))
    owns_land = rng.binomial(1, 0.55 - 0.20 * urban.astype(float))
    land_acres = np.where(owns_land, np.clip(rng.lognormal(0.5, 0.8, n), 0.1, 20), 0)
    owns_livestock = rng.binomial(1, 0.40 - 0.20 * urban.astype(float))

    # --- TRUE consumption (unobserved to targeter) ---
    # Log consumption model with realistic coefficients
    log_true_cons = (
        3.2
        + 0.07 * head_educ
        + 0.20 * urban.astype(float)
        - 0.08 * np.log(hh_size)
        + 0.15 * head_employed
        + 0.25 * wall_permanent
        + 0.12 * has_electricity
        + 0.10 * has_piped_water
        + 0.08 * owns_tv
        + 0.05 * owns_mobile
        + 0.06 * owns_motorcycle
        + 0.04 * np.log(land_acres + 1)
        + rng.normal(0, 0.45, n)  # large residual = targeting error
    )
    true_monthly_pce = np.round(np.exp(log_true_cons), 2)

    # --- PMT SCORE (predicted consumption from observable proxies) ---
    # PMT uses a subset of variables with estimated coefficients (+ noise)
    pmt_score = (
        3.1
        + 0.065 * head_educ
        + 0.18 * urban.astype(float)
        - 0.075 * np.log(hh_size)
        + 0.12 * head_employed
        + 0.22 * wall_permanent
        + 0.10 * has_electricity
        + 0.08 * has_piped_water
        + 0.07 * owns_tv
        + 0.04 * owns_mobile
        + 0.05 * owns_motorcycle
        + 0.03 * np.log(land_acres + 1)
        # No residual — PMT only uses fitted values
    )
    predicted_pce = np.round(np.exp(pmt_score), 2)

    # --- Poverty classification ---
    poverty_line = 65.0  # USD/month/capita
    truly_poor = (true_monthly_pce < poverty_line).astype(int)
    pmt_poor = (predicted_pce < poverty_line).astype(int)

    # Errors
    exclusion_error = (truly_poor & ~pmt_poor.astype(bool)).astype(int)
    inclusion_error = (~truly_poor.astype(bool) & pmt_poor).astype(int)

    # --- Community-based targeting ---
    # Village leaders rank households — noisy but captures some local info
    community_rank_score = (
        -log_true_cons
        + 0.3 * head_female
        + 0.2 * head_disabled
        + rng.normal(0, 0.6, n)  # leader bias/noise
    )
    community_selected = (community_rank_score > np.percentile(community_rank_score, 70)).astype(int)

    # --- Categorical targeting ---
    cat_eligible_elderly = (head_age >= 65).astype(int)
    cat_eligible_fhh = head_female
    cat_eligible_disabled = head_disabled

    # --- Benefit calculation ---
    # PMT-eligible get benefit (tapered by predicted consumption)
    monthly_benefit = np.where(
        pmt_poor,
        np.clip(poverty_line - predicted_pce, 5, poverty_line) * 0.5,
        0
    )
    monthly_benefit = np.round(monthly_benefit, 2)

    df = pd.DataFrame({
        "household_id": hh_ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "household_size": hh_size,
        "n_children_under15": n_children,
        "head_female": head_female,
        "head_age": head_age,
        "head_education_years": head_educ,
        "head_employed": head_employed,
        "head_disabled": head_disabled,
        "wall_permanent": wall_permanent,
        "roof_permanent": roof_permanent,
        "rooms": rooms,
        "has_electricity": has_electricity,
        "has_piped_water": has_piped_water,
        "has_flush_toilet": has_flush_toilet,
        "owns_radio": owns_radio,
        "owns_tv": owns_tv,
        "owns_mobile": owns_mobile,
        "owns_bicycle": owns_bicycle,
        "owns_motorcycle": owns_motorcycle,
        "owns_land": owns_land,
        "land_acres": np.round(land_acres, 2),
        "owns_livestock": owns_livestock,
        "true_monthly_pce_usd": true_monthly_pce,
        "pmt_predicted_pce_usd": predicted_pce,
        "poverty_line_usd": poverty_line,
        "truly_poor": truly_poor,
        "pmt_classified_poor": pmt_poor,
        "exclusion_error": exclusion_error,
        "inclusion_error": inclusion_error,
        "community_selected": community_selected,
        "categorical_eligible_elderly": cat_eligible_elderly,
        "categorical_eligible_fhh": cat_eligible_fhh,
        "categorical_eligible_disabled": cat_eligible_disabled,
        "monthly_benefit_usd": monthly_benefit,
    })

    df = inject_missing(df,
        columns=["head_education_years", "land_acres", "true_monthly_pce_usd"],
        rates=[0.03, 0.04, 0.10],
        rng=rng, mechanism="MNAR")
    return df
