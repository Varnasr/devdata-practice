"""
Generator 4: Agricultural / Crop Survey
────────────────────────────────────────
Simulates a plot-level agricultural survey with crop production, input use,
weather shocks, and market access — the kind of data used in studies of
agricultural productivity and technology adoption.

Rows: one per plot (~30-50k plots from ~15k households, 1-4 plots each).

Realistic features:
  • Cobb-Douglas production function with realistic elasticities
  • Correlated input use (fertilizer ↔ improved seed ↔ irrigation)
  • Weather shock (rainfall deviation) affecting yields
  • Distance to market affects output prices
  • Plot size in acres with skewed distribution
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_households: int = 15000, seed: int = 456) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Households
    hh_ids = household_ids(rng, n_households, prefix="FRM")
    districts, urban = pick_districts(rng, n_households, urban_share=0.15)
    n_plots = rng.choice([1, 2, 3, 4], n_households, p=[0.35, 0.35, 0.20, 0.10])
    countries = rng.choice(list(COUNTRIES.keys()), n_households)

    # Household-level: farmer characteristics
    head_age = rng.integers(20, 72, n_households)
    head_female = rng.binomial(1, 0.24, n_households)
    head_educ = np.clip(rng.normal(5, 3.5, n_households), 0, 16).astype(int)
    hh_size = rng.choice(range(2, 10), n_households)

    # Extension contact (correlated with education)
    extension_contact = rng.binomial(1, _logistic(0.25, (head_educ - 5) / 3, 0.6))

    # Wealth proxy
    wealth = rng.normal(0, 1, n_households) + 0.3 * head_educ / 10

    rows = []
    plot_counter = 1
    for i in range(n_households):
        # Distance to market (km)
        dist_market = np.clip(rng.exponential(12) + (0 if urban[i] else 10), 0.5, 80)

        for p in range(n_plots[i]):
            plot_id = f"PLT-{plot_counter:06d}"
            plot_counter += 1

            # Plot size (acres, right-skewed)
            plot_size = np.clip(rng.lognormal(0.3, 0.7), 0.1, 15)

            # Crop type
            crop = rng.choice(
                ["maize", "rice", "beans", "cassava", "sorghum", "groundnuts", "millet"],
                p=[0.30, 0.15, 0.15, 0.12, 0.10, 0.10, 0.08]
            )

            # Input use (correlated decisions)
            improved_seed = rng.binomial(1, _logistic(0.30, wealth[i], 0.5))
            fertilizer_kg = 0.0
            if rng.random() < _logistic(0.35, wealth[i], 0.5):
                fertilizer_kg = np.clip(rng.lognormal(2.5, 0.8), 5, 300)
            irrigation = rng.binomial(1, _logistic(0.10, wealth[i], 0.4))
            pesticide = rng.binomial(1, _logistic(0.20, wealth[i], 0.4))
            hired_labor_days = max(0, rng.poisson(max(0.1, 3 + 2 * wealth[i])))
            family_labor_days = rng.poisson(max(1, 8 + hh_size[i]))

            # Rainfall shock (deviation from normal, -2 to +2 SD)
            rainfall_deviation = rng.normal(0, 1)

            # Soil quality (1-5)
            soil_quality = rng.choice([1, 2, 3, 4, 5], p=[0.08, 0.18, 0.35, 0.25, 0.14])

            # Cobb-Douglas production: Y = A * L^a * F^b * S^c * R
            total_labor = hired_labor_days + family_labor_days
            A = _crop_tfp(crop) * (1 + 0.15 * improved_seed) * (1 + 0.10 * irrigation)
            A *= (0.8 + 0.1 * soil_quality)  # soil factor
            A *= np.exp(0.15 * rainfall_deviation - 0.10 * rainfall_deviation ** 2)  # inverted U

            log_yield = (
                np.log(A)
                + 0.50 * np.log(plot_size)
                + 0.25 * np.log(max(total_labor, 1))
                + 0.15 * np.log(max(fertilizer_kg, 1))
                + rng.normal(0, 0.3)
            )
            harvest_kg = np.clip(np.exp(log_yield), 5, 8000)

            # Output price (USD/kg, varies by crop and distance to market)
            base_price = _crop_price(crop)
            price_per_kg = base_price * (1 - 0.004 * dist_market) * rng.lognormal(0, 0.1)

            # Revenue
            revenue_usd = harvest_kg * max(price_per_kg, 0.01)

            # Input costs
            fert_cost = fertilizer_kg * rng.uniform(0.4, 0.8)
            seed_cost = plot_size * (rng.uniform(8, 20) if improved_seed else rng.uniform(2, 6))
            labor_cost = hired_labor_days * rng.uniform(2, 5)
            total_cost = fert_cost + seed_cost + labor_cost

            # Crop loss
            crop_loss_pct = 0
            if rainfall_deviation < -1.5 or rainfall_deviation > 1.8:
                crop_loss_pct = rng.uniform(10, 50)
            elif rng.random() < 0.08:  # pest/disease
                crop_loss_pct = rng.uniform(5, 30)

            rows.append({
                "plot_id": plot_id,
                "household_id": hh_ids[i],
                "country": countries[i],
                "district": districts[i],
                "urban": int(urban[i]),
                "head_age": head_age[i],
                "head_female": head_female[i],
                "head_education_years": head_educ[i],
                "household_size": hh_size[i],
                "extension_contact": extension_contact[i],
                "distance_to_market_km": round(dist_market, 1),
                "plot_size_acres": round(plot_size, 2),
                "crop": crop,
                "improved_seed": improved_seed,
                "fertilizer_kg": round(fertilizer_kg, 1),
                "irrigation": irrigation,
                "pesticide_used": pesticide,
                "hired_labor_days": hired_labor_days,
                "family_labor_days": family_labor_days,
                "soil_quality": soil_quality,
                "rainfall_deviation_sd": round(rainfall_deviation, 2),
                "harvest_kg": round(harvest_kg, 1),
                "crop_loss_pct": round(crop_loss_pct, 1),
                "price_per_kg_usd": round(price_per_kg, 3),
                "revenue_usd": round(revenue_usd, 2),
                "total_input_cost_usd": round(total_cost, 2),
                "profit_usd": round(revenue_usd - total_cost, 2),
            })

    df = pd.DataFrame(rows)
    df = inject_missing(df,
        columns=["fertilizer_kg", "harvest_kg", "revenue_usd", "soil_quality"],
        rates=[0.05, 0.04, 0.06, 0.10],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(base / (1 - base + 1e-9)) + slope * z)


def _crop_tfp(crop):
    return {"maize": 80, "rice": 90, "beans": 30, "cassava": 120,
            "sorghum": 50, "groundnuts": 25, "millet": 40}.get(crop, 50)


def _crop_price(crop):
    return {"maize": 0.22, "rice": 0.35, "beans": 0.55, "cassava": 0.10,
            "sorghum": 0.20, "groundnuts": 0.60, "millet": 0.25}.get(crop, 0.25)
