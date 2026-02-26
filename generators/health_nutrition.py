"""
Generator 5: Health & Nutrition Survey (DHS-style)
──────────────────────────────────────────────────
Simulates a Demographic and Health Survey with maternal health, child
anthropometrics, vaccination records, and reproductive health indicators.

Rows: one per child under 5 (with mother characteristics), ~30-40k rows.

Realistic features:
  • HAZ/WAZ/WHZ z-scores from WHO growth standards (correlated with wealth)
  • Vaccination schedule with age-appropriate coverage
  • Birth spacing and parity effects
  • Wealth quintile driving health access
  • Interviewer effects / heaping on age
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_children: int = 35000, seed: int = 789) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_children

    # --- IDs and geography ---
    child_ids = [f"CH-{i:06d}" for i in range(1, n + 1)]
    mother_ids = [f"MO-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.28)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # --- Wealth quintile (1-5) ---
    wealth_score = rng.normal(0, 1, n) + 0.5 * urban.astype(float)
    wealth_quintile = pd.qcut(wealth_score, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # --- Mother characteristics ---
    mother_age = rng.integers(15, 49, n)
    mother_educ = np.clip(
        rng.normal(4 + 1.5 * (wealth_quintile - 1), 2.5, n), 0, 16
    ).astype(int)
    parity = np.clip(rng.poisson(3.5 - 0.4 * (wealth_quintile - 1)), 1, 12)
    anc_visits = np.clip(
        rng.poisson(2.5 + 0.8 * (wealth_quintile - 1) + urban.astype(float)), 0, 12
    )
    facility_delivery = rng.binomial(
        1, _logistic(0.45, (wealth_quintile - 3) / 2, 0.7)
    )
    skilled_attendant = np.where(
        facility_delivery, 1,
        rng.binomial(1, _logistic(0.20, (wealth_quintile - 3) / 2, 0.5))
    )

    # --- Child characteristics ---
    child_age_months = rng.integers(0, 60, n)
    child_female = rng.binomial(1, 0.49, n)
    birth_weight_kg = np.clip(
        rng.normal(3.1 + 0.1 * (wealth_quintile - 1), 0.45, n), 1.0, 5.5
    )
    low_birth_weight = (birth_weight_kg < 2.5).astype(int)
    birth_order = np.clip(rng.poisson(2), 1, 10)

    # --- Anthropometrics (WHO z-scores) ---
    # Height-for-age (stunting if < -2)
    haz = (
        -1.0
        + 0.25 * (wealth_quintile - 3)
        + 0.15 * mother_educ / 10
        - 0.02 * np.maximum(child_age_months - 6, 0) / 12  # faltering
        + 0.3 * facility_delivery
        + rng.normal(0, 0.8, n)
    )
    haz = np.round(np.clip(haz, -6, 6), 2)
    stunted = (haz < -2).astype(int)
    severely_stunted = (haz < -3).astype(int)

    # Weight-for-age (underweight if < -2)
    waz = haz * 0.7 + rng.normal(0, 0.5, n)
    waz = np.round(np.clip(waz, -6, 5), 2)
    underweight = (waz < -2).astype(int)

    # Weight-for-height (wasting if < -2)
    whz = rng.normal(-0.3 + 0.1 * (wealth_quintile - 3), 1.0, n)
    whz = np.round(np.clip(whz, -5, 5), 2)
    wasted = (whz < -2).astype(int)

    # --- Feeding practices ---
    exclusive_bf_6m = np.where(
        child_age_months <= 6,
        rng.binomial(1, _logistic(0.45, (wealth_quintile - 3) / 2, 0.4)),
        np.nan  # not applicable
    )
    minimum_dietary_diversity = np.where(
        child_age_months >= 6,
        rng.binomial(1, _logistic(0.30, (wealth_quintile - 3) / 2, 0.5)),
        np.nan
    )

    # --- Vaccination (age-appropriate) ---
    bcg = np.where(child_age_months >= 1,
                   rng.binomial(1, _logistic(0.85, (wealth_quintile - 3) / 2, 0.3)), 0)
    dpt1 = np.where(child_age_months >= 2,
                    rng.binomial(1, _logistic(0.82, (wealth_quintile - 3) / 2, 0.3)), 0)
    dpt3 = np.where(child_age_months >= 4,
                    rng.binomial(1, np.clip(dpt1 * _logistic(0.70, (wealth_quintile - 3) / 2, 0.3), 0, 1)), 0)
    measles = np.where(child_age_months >= 9,
                       rng.binomial(1, _logistic(0.72, (wealth_quintile - 3) / 2, 0.3)), 0)
    fully_vaccinated = ((bcg == 1) & (dpt3 == 1) & (measles == 1)).astype(int)
    fully_vaccinated = np.where(child_age_months >= 12, fully_vaccinated, np.nan)

    # --- Morbidity (last 2 weeks) ---
    had_diarrhea = rng.binomial(1, _logistic(0.18, -(wealth_quintile - 3) / 2, 0.3))
    had_fever = rng.binomial(1, _logistic(0.22, -(wealth_quintile - 3) / 2, 0.3))
    had_cough = rng.binomial(1, _logistic(0.20, -(wealth_quintile - 3) / 2, 0.2))
    sought_treatment = np.where(
        (had_diarrhea | had_fever | had_cough),
        rng.binomial(1, _logistic(0.55, (wealth_quintile - 3) / 2, 0.4)),
        0
    )

    # Age heaping (interviewers round to 6, 12, 24, 36, 48)
    heap_mask = rng.random(n) < 0.12
    heaped_age = np.copy(child_age_months)
    for target in [6, 12, 24, 36, 48]:
        close = heap_mask & (np.abs(child_age_months - target) <= 2)
        heaped_age[close] = target

    df = pd.DataFrame({
        "child_id": child_ids,
        "mother_id": mother_ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "wealth_quintile": wealth_quintile,
        "mother_age": mother_age,
        "mother_education_years": mother_educ,
        "parity": parity,
        "anc_visits": anc_visits,
        "facility_delivery": facility_delivery,
        "skilled_birth_attendant": skilled_attendant,
        "child_age_months": heaped_age,
        "child_female": child_female,
        "birth_weight_kg": np.round(birth_weight_kg, 2),
        "low_birth_weight": low_birth_weight,
        "birth_order": birth_order,
        "height_for_age_z": haz,
        "weight_for_age_z": waz,
        "weight_for_height_z": whz,
        "stunted": stunted,
        "underweight": underweight,
        "wasted": wasted,
        "exclusive_bf_under6m": exclusive_bf_6m,
        "min_dietary_diversity": minimum_dietary_diversity,
        "bcg_vaccine": bcg,
        "dpt1_vaccine": dpt1,
        "dpt3_vaccine": dpt3,
        "measles_vaccine": measles,
        "fully_vaccinated": fully_vaccinated,
        "diarrhea_2wk": had_diarrhea,
        "fever_2wk": had_fever,
        "cough_2wk": had_cough,
        "sought_treatment": sought_treatment,
    })

    df = inject_missing(df,
        columns=["birth_weight_kg", "height_for_age_z", "weight_for_age_z",
                 "anc_visits", "mother_education_years"],
        rates=[0.15, 0.04, 0.04, 0.03, 0.02],
        rng=rng, mechanism="MAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(base / (1 - base + 1e-9)) + slope * np.asarray(z))
