"""
Generator: Environmental Justice & Pollution
─────────────────────────────────────────────
Simulates household-level data on environmental health, pollution exposure,
and distributive justice — the kind used in environmental justice research
and climate equity programmes.

Rows: ~20k households.

Realistic features:
  • Pollution exposure: air (PM2.5), indoor air, water, soil, noise
  • Environmental hazards: proximity to industrial sites and waste dumps
  • Health outcomes linked to pollution exposure
  • Cooking fuel type and location
  • Environmental assets: green space, tree cover, biodiversity
  • Climate justice: carbon footprint, vulnerability, displacement
  • Environmental governance: rights awareness, complaints
  • Environmental racism/classism: marginalized communities face higher exposure
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_households: int = 20000, seed: int = 806) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    hh_ids = household_ids(rng, n)
    districts, urban = pick_districts(rng, n, urban_share=0.38)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    hh_size = rng.choice(range(1, 11), n,
                         p=[0.03, 0.06, 0.10, 0.16, 0.22, 0.17, 0.12, 0.07, 0.04, 0.03])
    head_female = rng.binomial(1, 0.26, n)
    head_education_years = np.clip(rng.normal(5.5 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    wealth_quintile = rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.21, 0.20, 0.19, 0.18])
    wealth = (wealth_quintile - 3) / 2.0 + rng.normal(0, 0.3, n)

    # Marginalization proxy (poorer and less educated = more marginalized)
    marginalized = (wealth_quintile <= 2).astype(float)

    # --- Pollution exposure ---
    # Environmental racism/classism: poorer communities have higher pollution
    air_quality_pm25 = np.clip(
        rng.normal(45 - 8 * wealth + 10 * urban.astype(float) + 8 * marginalized, 15, n),
        5, 200
    ).round(1)

    fuels = ["firewood", "charcoal", "kerosene", "lpg", "electricity", "biogas"]
    fuel_probs_poor = [0.35, 0.20, 0.15, 0.18, 0.05, 0.07]
    fuel_probs_rich = [0.05, 0.05, 0.08, 0.50, 0.25, 0.07]
    cooking_fuel_type = np.where(
        wealth_quintile >= 4,
        rng.choice(fuels, n, p=fuel_probs_rich),
        rng.choice(fuels, n, p=fuel_probs_poor)
    )

    cooking_location = rng.choice(
        ["indoor_no_vent", "indoor_vented", "outdoor"],
        n, p=[0.30, 0.40, 0.30]
    )
    dirty_fuel = np.isin(cooking_fuel_type, ["firewood", "charcoal", "kerosene"]).astype(float)
    indoor_no_vent = (cooking_location == "indoor_no_vent").astype(float)
    indoor_air_pollution = rng.binomial(1, _logistic(0.30, dirty_fuel * 0.6 + indoor_no_vent * 0.4, 0.5))

    water_contamination_score = np.clip(
        rng.normal(4 - 1.5 * wealth + 1.5 * marginalized, 2, n), 0, 10
    ).round(1)

    soil_contamination_risk = rng.binomial(1, _logistic(0.15, -wealth + 0.2 * marginalized, 0.4))
    noise_pollution_level = np.clip(
        rng.normal(55 + 10 * urban.astype(float) - 3 * wealth, 12, n), 20, 100
    ).round(1)

    # --- Environmental hazards ---
    proximity_to_industrial_site_km = np.clip(
        rng.exponential(5 - 1.5 * marginalized + 2 * (1 - urban.astype(float)), n),
        0.1, 50
    ).round(1)
    proximity_to_waste_dump_km = np.clip(
        rng.exponential(4 - 1.2 * marginalized, n),
        0.1, 40
    ).round(1)
    flood_risk_zone = rng.binomial(1, _logistic(0.18, -wealth + 0.1 * marginalized, 0.3))
    landslide_risk = rng.binomial(1, _logistic(0.08, -urban.astype(float) * 0.3, 0.3))

    # --- Health outcomes (linked to pollution exposure) ---
    pollution_load = (air_quality_pm25 / 100 + indoor_air_pollution * 0.3
                      + water_contamination_score / 10)
    respiratory_illness_12m = rng.binomial(1, _logistic(0.15, pollution_load, 0.5))
    waterborne_illness_12m = rng.binomial(1, _logistic(0.12, water_contamination_score / 5 - wealth * 0.2, 0.5))
    skin_condition = rng.binomial(1, _logistic(0.08, water_contamination_score / 8 + soil_contamination_risk * 0.3, 0.4))
    child_blood_lead_elevated = rng.binomial(
        1, _logistic(0.06, soil_contamination_risk * 0.4 + marginalized * 0.2 - wealth * 0.1, 0.5)
    )

    # --- Environmental assets ---
    access_to_green_space = rng.binomial(1, _logistic(0.40, wealth * 0.2 - urban.astype(float) * 0.15 + 0.1 * (1 - marginalized), 0.3))
    tree_cover_pct = np.clip(
        rng.normal(25 - 8 * urban.astype(float) + 3 * wealth, 12, n), 0, 80
    ).round(1)
    biodiversity_score = np.clip(
        rng.normal(5 - 1.5 * urban.astype(float) + 0.5 * wealth, 1.5, n), 0, 10
    ).round(1)

    # --- Climate justice ---
    carbon_footprint_tco2 = np.clip(
        rng.lognormal(np.log(1.5) + 0.3 * wealth + 0.2 * urban.astype(float), 0.4, n),
        0.2, 20
    ).round(2)
    climate_vulnerability_score = np.clip(
        rng.normal(5 - 1.5 * wealth + 1.0 * marginalized + 0.5 * flood_risk_zone, 1.5, n), 0, 10
    ).round(1)
    experienced_environmental_displacement = rng.binomial(
        1, _logistic(0.08, flood_risk_zone * 0.3 + marginalized * 0.2 - wealth * 0.1, 0.4)
    )

    # --- Governance ---
    aware_of_environmental_rights = rng.binomial(
        1, _logistic(0.25, 0.1 * head_education_years / 10 + 0.15 * urban.astype(float), 0.4)
    )
    filed_environmental_complaint = rng.binomial(
        1, _logistic(0.08, aware_of_environmental_rights * 0.3 + wealth * 0.1, 0.4)
    )
    complaint_resolved = np.where(
        filed_environmental_complaint,
        rng.binomial(1, 0.30 + 0.10 * wealth),
        0
    )

    df = pd.DataFrame({
        "household_id": hh_ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "household_size": hh_size,
        "head_female": head_female, "head_education_years": head_education_years,
        "wealth_quintile": wealth_quintile,
        "air_quality_pm25": air_quality_pm25,
        "indoor_air_pollution": indoor_air_pollution,
        "water_contamination_score": water_contamination_score,
        "soil_contamination_risk": soil_contamination_risk,
        "noise_pollution_level": noise_pollution_level,
        "proximity_to_industrial_site_km": proximity_to_industrial_site_km,
        "proximity_to_waste_dump_km": proximity_to_waste_dump_km,
        "flood_risk_zone": flood_risk_zone,
        "landslide_risk": landslide_risk,
        "respiratory_illness_12m": respiratory_illness_12m,
        "waterborne_illness_12m": waterborne_illness_12m,
        "skin_condition": skin_condition,
        "child_blood_lead_elevated": child_blood_lead_elevated,
        "cooking_fuel_type": cooking_fuel_type,
        "cooking_location": cooking_location,
        "access_to_green_space": access_to_green_space,
        "tree_cover_pct": tree_cover_pct,
        "biodiversity_score": biodiversity_score,
        "carbon_footprint_tco2": carbon_footprint_tco2,
        "climate_vulnerability_score": climate_vulnerability_score,
        "experienced_environmental_displacement": experienced_environmental_displacement,
        "aware_of_environmental_rights": aware_of_environmental_rights,
        "filed_environmental_complaint": filed_environmental_complaint,
        "complaint_resolved": complaint_resolved,
    })

    df = inject_missing(df,
        columns=["air_quality_pm25", "water_contamination_score", "tree_cover_pct", "noise_pollution_level"],
        rates=[0.05, 0.07, 0.04, 0.06],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
