"""
Generator 1: Household Survey (LSMS-style)
──────────────────────────────────────────
Produces a multi-module household survey resembling the World Bank's Living
Standards Measurement Study.  Modules: demographics, consumption/expenditure,
assets, housing, food security, and subjective well-being.

Rows: one per household member (long format) — typically 50-100k rows from
~10-20k households with 2-8 members each.

Realistic features:
  • Intra-household correlation (shared district, wealth, housing)
  • Log-normal consumption with Engel-curve food shares
  • Asset index via PCA-like correlated binary ownership
  • MNAR missingness (richer HH less likely to report income)
"""

import numpy as np
import pandas as pd
from .utils import (
    household_ids, individual_ids, pick_districts, inject_missing,
    correlated_normal, random_dates, COUNTRIES,
)


def generate(n_households: int = 15000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # --- Household-level scaffold ---
    hh_ids = household_ids(rng, n_households)
    hh_size = rng.choice([2, 3, 4, 5, 6, 7, 8],
                         size=n_households,
                         p=[0.05, 0.12, 0.22, 0.25, 0.18, 0.12, 0.06])
    districts, urban = pick_districts(rng, n_households, urban_share=0.32)
    countries = rng.choice(list(COUNTRIES.keys()), size=n_households,
                           p=_country_weights(len(COUNTRIES)))
    survey_date = random_dates(rng, n_households, "2022-01-01", "2023-06-30")

    # Household-level latent wealth (drives many variables)
    wealth_latent = rng.normal(0, 1, n_households)
    wealth_latent[urban] += 0.6  # urban premium

    # Monthly per-capita consumption (USD PPP, log-normal)
    log_cons = 3.8 + 0.5 * wealth_latent + rng.normal(0, 0.4, n_households)
    monthly_pce = np.exp(log_cons)

    # Food share (Engel curve — declines with wealth)
    food_share = np.clip(0.70 - 0.08 * wealth_latent + rng.normal(0, 0.06, n_households), 0.20, 0.90)

    # Asset ownership (correlated binary)
    asset_names = ["radio", "tv", "mobile_phone", "bicycle", "motorcycle",
                   "refrigerator", "solar_panel", "improved_stove"]
    asset_probs_base = np.array([0.60, 0.30, 0.75, 0.40, 0.15, 0.12, 0.18, 0.25])
    assets = {}
    for i, name in enumerate(asset_names):
        p = _logistic(asset_probs_base[i], wealth_latent, slope=0.8)
        assets[f"owns_{name}"] = rng.binomial(1, p)

    # Housing
    wall_material = np.where(
        rng.random(n_households) < _logistic(0.4, wealth_latent, 0.7),
        "permanent", np.where(rng.random(n_households) < 0.5, "semi-permanent", "temporary")
    )
    rooms = np.clip(rng.poisson(np.clip(1.5 + 0.5 * wealth_latent, 0.1, 10)), 1, 8)
    water_source = np.where(
        rng.random(n_households) < _logistic(0.45, wealth_latent, 0.6),
        "piped", np.where(rng.random(n_households) < 0.4, "borehole", "surface")
    )
    toilet_type = np.where(
        rng.random(n_households) < _logistic(0.30, wealth_latent, 0.7),
        "flush", np.where(rng.random(n_households) < 0.5, "pit_latrine", "none")
    )

    # Food security (FIES-like 0-8 scale)
    food_insecurity = np.clip(
        rng.poisson(np.clip(np.exp(1.2 - 0.5 * wealth_latent), 0.01, 50)), 0, 8
    )

    # Subjective well-being (1-10)
    life_satisfaction = np.clip(
        np.round(5.5 + 1.0 * wealth_latent + rng.normal(0, 1.2, n_households)), 1, 10
    ).astype(int)

    # --- Expand to individual level ---
    rows = []
    ind_counter = 1
    for i in range(n_households):
        n_members = hh_size[i]
        head_age = rng.integers(22, 70)
        head_female = int(rng.random() < 0.28)
        head_educ = _education_years(rng, wealth_latent[i], urban[i])

        for m in range(n_members):
            is_head = m == 0
            if is_head:
                age = head_age
                female = head_female
                educ = head_educ
                relationship = "head"
            else:
                relationship = rng.choice(
                    ["spouse", "child", "child", "child", "other_relative"],
                )
                if relationship == "spouse":
                    age = max(18, head_age + rng.integers(-5, 5))
                    female = 1 - head_female
                    educ = _education_years(rng, wealth_latent[i], urban[i])
                elif relationship == "child":
                    age = max(0, head_age - rng.integers(15, 35))
                    female = int(rng.random() < 0.50)
                    educ = _education_years(rng, wealth_latent[i], urban[i]) if age >= 6 else 0
                else:
                    age = rng.integers(10, 80)
                    female = int(rng.random() < 0.52)
                    educ = _education_years(rng, wealth_latent[i], urban[i]) if age >= 6 else 0

                # Cap education by age
                educ = min(educ, max(0, age - 5))

            rows.append({
                "individual_id": f"IND-{ind_counter:07d}",
                "household_id": hh_ids[i],
                "country": countries[i],
                "district": districts[i],
                "urban": int(urban[i]),
                "survey_date": survey_date[i],
                "relationship": relationship,
                "age": age,
                "female": female,
                "education_years": educ,
                "household_size": n_members,
                "monthly_pce_usd": round(monthly_pce[i], 2),
                "food_share": round(food_share[i], 3),
                "food_insecurity_score": food_insecurity[i],
                "life_satisfaction": life_satisfaction[i],
                "wall_material": wall_material[i],
                "rooms": rooms[i],
                "water_source": water_source[i],
                "toilet_type": toilet_type[i],
                **{k: v[i] for k, v in assets.items()},
            })
            ind_counter += 1

    df = pd.DataFrame(rows)

    # Inject realistic missingness
    df = inject_missing(df,
        columns=["monthly_pce_usd", "education_years", "food_share",
                 "life_satisfaction", "food_insecurity_score"],
        rates=[0.06, 0.03, 0.08, 0.04, 0.05],
        rng=rng, mechanism="MNAR")

    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _logistic(base_prob, z, slope=1.0):
    """Shift a base probability by a latent z through a logistic link."""
    from scipy.special import expit
    logit_base = np.log(base_prob / (1 - base_prob + 1e-9))
    return expit(logit_base + slope * z)


def _education_years(rng, wealth_z, is_urban):
    base = 6 + 2.5 * wealth_z + (2 if is_urban else 0)
    return int(np.clip(base + rng.normal(0, 2), 0, 18))


def _country_weights(n):
    w = np.ones(n)
    w /= w.sum()
    return w
