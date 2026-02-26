"""
Generator: Unpaid Care & Time Use
─────────────────────────────────
Simulates time use diary data capturing unpaid care work, domestic labour,
paid work, and leisure, with a strong gender dimension.

Rows: ~20k individuals (mixed gender).

Realistic features:
  • Time allocations that roughly sum to 24 hours/day
  • Women do ~3x more unpaid care than men
  • Care infrastructure reduces unpaid care burden
  • Care breakdown (childcare, eldercare, cooking, cleaning, water, fuel)
  • Opportunity cost and time poverty measures
  • MCAR missingness mechanism
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_individuals: int = 20000, seed: int = 804) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    # --- Demographics ---
    ids = [f"IND-{i:07d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.35)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.52, n)
    age = rng.integers(15, 70, n)
    educ_years = np.clip(
        rng.normal(7 + 2 * urban.astype(float), 3.5, n), 0, 18
    ).astype(int)
    married = rng.binomial(1, np.where(age < 20, 0.05, np.where(age < 30, 0.40, 0.70)))
    n_children = np.clip(
        rng.poisson(np.clip(1.8 + 0.5 * married - 0.04 * educ_years + 0.3 * (1 - urban.astype(float)), 0.1, 10), n),
        0, 8
    )
    n_elderly_in_hh = np.clip(rng.poisson(np.clip(0.3 + 0.1 * (1 - urban.astype(float)), 0.05, 5), n), 0, 3)
    wealth_latent = rng.normal(0, 1, n) + 0.3 * urban.astype(float) + 0.05 * educ_years
    wealth_quintile = np.clip(
        (pd.qcut(wealth_latent, 5, labels=False, duplicates="drop") + 1), 1, 5
    ).astype(int)

    # --- Care infrastructure ---
    distance_to_water_min = np.clip(
        rng.lognormal(2.5 - 1.0 * urban.astype(float), 0.8, n), 0, 120
    ).round(0).astype(int)
    distance_to_health_facility_km = np.clip(
        rng.lognormal(1.5 - 0.6 * urban.astype(float), 0.7, n), 0.2, 50
    ).round(1)
    has_childcare_access = rng.binomial(
        1, _logistic(0.20, 0.3 * urban.astype(float) + 0.1 * wealth_latent, 0.3)
    )
    has_electricity = rng.binomial(
        1, _logistic(0.45, 0.4 * urban.astype(float) + 0.15 * wealth_latent, 0.4)
    )
    has_improved_cookstove = rng.binomial(
        1, _logistic(0.20, 0.2 * urban.astype(float) + 0.1 * wealth_latent, 0.3)
    )

    # Infrastructure composite for reducing care burden
    infra_relief = (
        0.15 * has_childcare_access
        + 0.10 * has_electricity
        + 0.08 * has_improved_cookstove
        + 0.10 * urban.astype(float)
    )

    # --- Time use allocations (hours/day) ---
    # Sleep
    sleep_hours = np.clip(rng.normal(7.5, 0.8, n), 5, 10).round(1)

    # Personal care
    personal_care_hours = np.clip(rng.normal(1.5, 0.4, n), 0.5, 3.0).round(1)

    # Paid work: men more, educated more, urban more
    paid_work_base = (
        4.0
        - 2.5 * female  # men work more paid hours
        + 0.8 * urban.astype(float)
        + 0.1 * educ_years
        - 0.5 * (n_children > 2).astype(float) * female  # mothers with many kids work less
    )
    paid_work_hours = np.clip(rng.normal(paid_work_base, 2.0, n), 0, 12).round(1)

    # Unpaid care: women ~3x men, more with children/elderly
    care_base_female = 3.5 + 0.5 * n_children + 0.4 * n_elderly_in_hh - infra_relief * 3
    care_base_male = 1.0 + 0.15 * n_children + 0.1 * n_elderly_in_hh - infra_relief * 1
    unpaid_care_base = np.where(female, care_base_female, care_base_male)
    unpaid_care_hours = np.clip(rng.normal(unpaid_care_base, 1.0, n), 0, 10).round(1)

    # Domestic work: women more, infrastructure reduces
    domestic_base_female = 2.5 - infra_relief * 2 + 0.2 * n_children
    domestic_base_male = 0.8 - infra_relief * 0.5
    domestic_base = np.where(female, domestic_base_female, domestic_base_male)
    domestic_work_hours = np.clip(rng.normal(domestic_base, 0.8, n), 0, 7).round(1)

    # Education hours (younger, more educated)
    educ_base = np.where(age < 25, 2.5, np.where(age < 35, 0.5, 0.1)) - 0.3 * married
    education_hours = np.clip(rng.normal(educ_base, 1.0, n), 0, 8).round(1)

    # Commute
    commute_hours = np.clip(
        rng.lognormal(-0.5 + 0.3 * urban.astype(float), 0.6, n), 0, 4
    ).round(1)

    # Leisure: residual to make ~24 hours
    allocated = (sleep_hours + personal_care_hours + paid_work_hours +
                 unpaid_care_hours + domestic_work_hours + education_hours + commute_hours)
    leisure_hours = np.clip(24.0 - allocated, 0.5, 8.0).round(1)

    # --- Care breakdown (sub-components of unpaid_care + domestic) ---
    total_care_domestic = unpaid_care_hours + domestic_work_hours
    # Proportional split
    childcare_share = np.clip(0.30 + 0.05 * n_children - 0.02 * n_elderly_in_hh, 0.05, 0.60)
    eldercare_share = np.clip(0.05 + 0.10 * n_elderly_in_hh, 0.0, 0.30)
    cooking_share = np.clip(0.25 - 0.03 * has_improved_cookstove, 0.10, 0.40)
    cleaning_share = np.clip(0.15, 0.05, 0.25)
    water_share = np.clip(0.08 * (1 - urban.astype(float)) * (distance_to_water_min / 60), 0.0, 0.15)
    fuel_share = np.clip(0.05 * (1 - has_electricity) * (1 - urban.astype(float)), 0.0, 0.10)

    # Normalize shares to sum to 1
    share_sum = childcare_share + eldercare_share + cooking_share + cleaning_share + water_share + fuel_share
    childcare_hours = np.round(total_care_domestic * childcare_share / share_sum, 1)
    eldercare_hours = np.round(total_care_domestic * eldercare_share / share_sum, 1)
    cooking_hours = np.round(total_care_domestic * cooking_share / share_sum, 1)
    cleaning_hours = np.round(total_care_domestic * cleaning_share / share_sum, 1)
    water_collection_hours = np.round(total_care_domestic * water_share / share_sum, 1)
    fuel_collection_hours = np.round(total_care_domestic * fuel_share / share_sum, 1)

    # --- Opportunity cost ---
    # Estimated hourly wage from education/age (potential earnings)
    potential_log_wage = 0.8 + 0.06 * educ_years + 0.02 * np.clip(age - educ_years - 6, 0, 40)
    potential_hourly = np.exp(potential_log_wage)
    forgone_earnings_usd = np.round(
        (unpaid_care_hours + domestic_work_hours) * potential_hourly * 30, 2
    )

    reduced_labor_participation = rng.binomial(
        1, _logistic(0.25, 0.3 * female * (unpaid_care_hours > 3).astype(float)
                     + 0.1 * (n_children > 2).astype(float), 0.4)
    )

    # --- Time poverty (>10.5 hours/day on paid + unpaid work) ---
    total_work = paid_work_hours + unpaid_care_hours + domestic_work_hours
    time_poor = (total_work > 10.5).astype(int)

    # --- Programme support ---
    received_care_support = rng.binomial(1, 0.12, n)
    care_support_types = ["childcare_subsidy", "community_creche", "cash_for_care",
                          "labor_saving_tech", "parental_leave", "none"]
    care_support_type = np.where(
        received_care_support,
        rng.choice(care_support_types[:5], n, p=[0.25, 0.20, 0.20, 0.20, 0.15]),
        "none",
    )

    df = pd.DataFrame({
        "individual_id": ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "education_years": educ_years,
        "married": married,
        "n_children": n_children,
        "n_elderly_in_hh": n_elderly_in_hh,
        "wealth_quintile": wealth_quintile,
        "sleep_hours": sleep_hours,
        "paid_work_hours": paid_work_hours,
        "unpaid_care_hours": unpaid_care_hours,
        "domestic_work_hours": domestic_work_hours,
        "education_hours": education_hours,
        "leisure_hours": leisure_hours,
        "personal_care_hours": personal_care_hours,
        "commute_hours": commute_hours,
        "childcare_hours": childcare_hours,
        "eldercare_hours": eldercare_hours,
        "cooking_hours": cooking_hours,
        "cleaning_hours": cleaning_hours,
        "water_collection_hours": water_collection_hours,
        "fuel_collection_hours": fuel_collection_hours,
        "distance_to_water_min": distance_to_water_min,
        "distance_to_health_facility_km": distance_to_health_facility_km,
        "has_childcare_access": has_childcare_access,
        "has_electricity": has_electricity,
        "has_improved_cookstove": has_improved_cookstove,
        "forgone_earnings_usd": forgone_earnings_usd,
        "reduced_labor_participation": reduced_labor_participation,
        "time_poor": time_poor,
        "received_care_support": received_care_support,
        "care_support_type": care_support_type,
    })

    df = inject_missing(df,
        columns=["unpaid_care_hours", "paid_work_hours", "forgone_earnings_usd",
                 "leisure_hours", "cooking_hours"],
        rates=[0.05, 0.04, 0.07, 0.03, 0.04],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + slope * np.asarray(z))
