"""
Generator: Climate & Resilience
───────────────────────────────
Simulates household-level climate vulnerability, adaptation, shock exposure,
and resilience capacity data — the kind used in climate adaptation programmes
and resilience measurement (RIMA, PRIME, AbFM).

Rows: ~20k households.

Realistic features:
  • Climate shock exposure (drought, flood, cyclone) with geographic clustering
  • Coping strategies (consumption smoothing, asset depletion, migration)
  • Adaptive capacity index (RIMA-like pillars)
  • Livelihood diversification score
  • Access to climate information and early warning
  • Carbon footprint / emissions proxy at household level
  • Before/after resilience measurement
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_households: int = 20000, seed: int = 703) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    hh_ids = household_ids(rng, n)
    districts, urban = pick_districts(rng, n, urban_share=0.25)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    hh_size = rng.choice(range(1, 11), n,
                         p=[0.02, 0.05, 0.10, 0.15, 0.22, 0.18, 0.13, 0.08, 0.05, 0.02])
    head_female = rng.binomial(1, 0.28, n)
    head_age = rng.integers(20, 72, n)
    head_educ = np.clip(rng.normal(5.5, 3.5, n), 0, 18).astype(int)

    # Agro-ecological zone
    aez = rng.choice(["arid", "semi_arid", "sub_humid", "humid", "highland"],
                     n, p=[0.15, 0.25, 0.30, 0.18, 0.12])

    # Wealth & livelihoods
    wealth = rng.normal(0, 1, n) + 0.3 * urban.astype(float) + 0.2 * head_educ / 10
    primary_livelihood = rng.choice(
        ["crop_farming", "livestock", "mixed_farming", "wage_labor", "fishing",
         "small_business", "casual_labor"],
        n, p=[0.25, 0.12, 0.18, 0.15, 0.05, 0.12, 0.13]
    )
    n_income_sources = np.clip(rng.poisson(2 + 0.3 * wealth), 1, 6)
    livelihood_diversification = n_income_sources / 6

    # --- Climate shock exposure (last 3 years) ---
    drought_zone = np.isin(aez, ["arid", "semi_arid"])
    flood_zone = np.isin(aez, ["sub_humid", "humid"])

    experienced_drought = rng.binomial(1, np.where(drought_zone, 0.55, 0.15))
    experienced_flood = rng.binomial(1, np.where(flood_zone, 0.40, 0.10))
    experienced_cyclone = rng.binomial(1, 0.08, n)
    experienced_pest_outbreak = rng.binomial(1, 0.18, n)
    experienced_disease_outbreak = rng.binomial(1, 0.12, n)
    n_shocks = experienced_drought + experienced_flood + experienced_cyclone + experienced_pest_outbreak + experienced_disease_outbreak
    any_shock = (n_shocks > 0).astype(int)

    # Shock severity (1-5, if experienced any shock)
    shock_severity = np.where(any_shock, np.clip(rng.poisson(2.5, n), 1, 5), 0)

    # --- Losses from shocks ---
    crop_loss_pct = np.where(any_shock, np.clip(rng.beta(2, 3, n) * 100, 0, 100), 0)
    livestock_loss_pct = np.where(any_shock & (primary_livelihood == "livestock"),
                                  np.clip(rng.beta(1.5, 4, n) * 100, 0, 80), 0)
    income_loss_pct = np.where(any_shock, np.clip(rng.beta(2, 4, n) * 100, 0, 80), 0)

    # --- Coping strategies ---
    cs_reduced_meals = any_shock * rng.binomial(1, _logistic(0.45, -wealth, 0.3))
    cs_sold_assets = any_shock * rng.binomial(1, _logistic(0.30, -wealth, 0.3))
    cs_borrowed_money = any_shock * rng.binomial(1, 0.35, n)
    cs_migration = any_shock * rng.binomial(1, _logistic(0.12, -wealth, 0.3))
    cs_reduced_education = any_shock * rng.binomial(1, _logistic(0.15, -wealth, 0.3))
    cs_sold_livestock = any_shock * rng.binomial(1, 0.20, n)

    coping_strategies_index = (cs_reduced_meals + cs_sold_assets + cs_borrowed_money +
                               cs_migration + cs_reduced_education + cs_sold_livestock)

    # --- Adaptation & preparedness ---
    adopted_drought_resistant_crop = rng.binomial(1, _logistic(0.20, wealth + 0.1 * head_educ / 10, 0.3))
    adopted_irrigation = rng.binomial(1, _logistic(0.12, wealth, 0.4))
    adopted_soil_conservation = rng.binomial(1, _logistic(0.18, wealth + 0.05 * head_educ / 10, 0.3))
    has_crop_insurance = rng.binomial(1, _logistic(0.08, wealth + 0.1 * urban.astype(float), 0.5))
    has_savings = rng.binomial(1, _logistic(0.30, wealth, 0.4))
    member_community_group = rng.binomial(1, _logistic(0.35, 0.1 * head_educ / 10, 0.3))

    # Early warning & climate info
    received_early_warning = rng.binomial(1, _logistic(0.35, urban.astype(float) * 0.3 + wealth * 0.1, 0.3))
    access_climate_info = rng.binomial(1, _logistic(0.30, 0.2 * urban.astype(float) + 0.1 * head_educ / 10, 0.3))

    # --- Resilience index (RIMA-like: 0-1) ---
    # Pillars: absorptive, adaptive, transformative
    absorptive = np.clip(
        0.3 * has_savings + 0.2 * has_crop_insurance + 0.2 * member_community_group
        + 0.15 * received_early_warning + 0.15 * (1 - cs_sold_assets)
        + rng.normal(0, 0.05, n), 0, 1
    )
    adaptive = np.clip(
        0.25 * livelihood_diversification + 0.20 * adopted_drought_resistant_crop
        + 0.20 * adopted_irrigation + 0.15 * adopted_soil_conservation
        + 0.10 * access_climate_info + 0.10 * (head_educ / 18)
        + rng.normal(0, 0.05, n), 0, 1
    )
    transformative = np.clip(
        0.30 * (head_educ / 18) + 0.25 * urban.astype(float)
        + 0.20 * has_savings + 0.15 * (1 - head_female)
        + 0.10 * member_community_group
        + rng.normal(0, 0.05, n), 0, 1
    )
    resilience_index = np.round(0.35 * absorptive + 0.35 * adaptive + 0.30 * transformative, 3)

    # --- Food security (post-shock) ---
    food_consumption_score = np.clip(
        rng.normal(55 + 10 * wealth - 8 * any_shock + 5 * has_savings, 12, n), 0, 112
    ).round(1)

    # Carbon footprint proxy (tCO2/year, lower for poor)
    carbon_footprint = np.clip(
        rng.lognormal(np.log(1.5) + 0.3 * wealth + 0.2 * urban.astype(float), 0.4, n),
        0.2, 15
    ).round(2)

    df = pd.DataFrame({
        "household_id": hh_ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "agroecological_zone": aez,
        "household_size": hh_size, "head_female": head_female,
        "head_age": head_age, "head_education_years": head_educ,
        "primary_livelihood": primary_livelihood,
        "n_income_sources": n_income_sources,
        "livelihood_diversification": np.round(livelihood_diversification, 3),
        "experienced_drought": experienced_drought,
        "experienced_flood": experienced_flood,
        "experienced_cyclone": experienced_cyclone,
        "experienced_pest_outbreak": experienced_pest_outbreak,
        "n_climate_shocks": n_shocks,
        "shock_severity": shock_severity,
        "crop_loss_pct": np.round(crop_loss_pct, 1),
        "livestock_loss_pct": np.round(livestock_loss_pct, 1),
        "income_loss_pct": np.round(income_loss_pct, 1),
        "cs_reduced_meals": cs_reduced_meals,
        "cs_sold_assets": cs_sold_assets,
        "cs_borrowed_money": cs_borrowed_money,
        "cs_migration": cs_migration,
        "cs_reduced_education": cs_reduced_education,
        "coping_strategies_index": coping_strategies_index,
        "adopted_drought_resistant_crop": adopted_drought_resistant_crop,
        "adopted_irrigation": adopted_irrigation,
        "adopted_soil_conservation": adopted_soil_conservation,
        "has_crop_insurance": has_crop_insurance,
        "has_savings": has_savings,
        "member_community_group": member_community_group,
        "received_early_warning": received_early_warning,
        "access_climate_info": access_climate_info,
        "absorptive_capacity": np.round(absorptive, 3),
        "adaptive_capacity": np.round(adaptive, 3),
        "transformative_capacity": np.round(transformative, 3),
        "resilience_index": resilience_index,
        "food_consumption_score": food_consumption_score,
        "carbon_footprint_tco2_yr": carbon_footprint,
    })

    df = inject_missing(df,
        columns=["crop_loss_pct", "income_loss_pct", "food_consumption_score", "carbon_footprint_tco2_yr"],
        rates=[0.08, 0.06, 0.04, 0.10],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
