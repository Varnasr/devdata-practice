"""
Generator 7: Labor Market / Employment Survey
──────────────────────────────────────────────
Simulates an individual-level labor force survey with employment status,
wages, sector, formality, hours worked, and migration history.

Rows: ~40k working-age adults (15-64).

Realistic features:
  • Mincer wage equation (returns to education + experience)
  • Formal/informal sector wage premium
  • Gender wage gap with Oaxaca-Blinder decomposition structure
  • Self-employment vs. wage employment vs. unemployed
  • Rural-urban migration and remittances
  • Seasonal employment variation
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_individuals: int = 40000, seed: int = 303) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    # Demographics
    ids = [f"WRK-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.38)
    countries = rng.choice(list(COUNTRIES.keys()), n)
    female = rng.binomial(1, 0.50, n)
    age = rng.integers(15, 65, n)

    # Education (correlated with urban)
    educ_years = np.clip(
        rng.normal(7 + 3 * urban.astype(float), 3.5, n), 0, 20
    ).astype(int)

    # Experience (Mincer: potential experience = age - educ - 6)
    experience = np.clip(age - educ_years - 6, 0, 45)

    # Marital status
    married = rng.binomial(1, np.where(age < 20, 0.05, np.where(age < 30, 0.4, 0.7)))

    # Migration status
    migrant = rng.binomial(1, 0.15 + 0.08 * urban.astype(float))
    sends_remittances = migrant * rng.binomial(1, 0.6, n)

    # Employment status
    emp_probs = np.column_stack([
        _logistic(0.10, -(educ_years - 5) / 4 + 0.3 * (age < 25).astype(float), 0.5),  # unemployed
        _logistic(0.35, (educ_years - 7) / 4 + 0.3 * urban.astype(float), 0.4),  # wage employed
        np.full(n, 0.30),  # self-employed
        np.full(n, 0.10),  # unpaid family worker
    ])
    # Normalize
    emp_probs = emp_probs / emp_probs.sum(axis=1, keepdims=True)
    emp_status = np.array([
        rng.choice(["unemployed", "wage_employed", "self_employed", "unpaid_family"],
                   p=emp_probs[i])
        for i in range(n)
    ])

    # Sector (for employed)
    sectors = ["agriculture", "manufacturing", "construction", "trade_retail",
               "transport", "services", "public_admin", "domestic_work"]
    sector = np.where(
        emp_status == "unemployed", "none",
        [rng.choice(sectors, p=_sector_probs(urban[i])) for i in range(n)]
    )

    # Formal vs informal
    formal = np.zeros(n, dtype=int)
    employed_mask = emp_status != "unemployed"
    formal[employed_mask] = rng.binomial(
        1, _logistic(0.25, (educ_years[employed_mask] - 8) / 3 + 0.4 * urban[employed_mask].astype(float), 0.6)
    )

    # Wages (Mincer equation, only for employed)
    # ln(wage) = β₀ + β₁*educ + β₂*exp + β₃*exp² + β₄*female + β₅*urban + β₆*formal + ε
    log_wage = np.full(n, np.nan)
    emask = employed_mask & (emp_status != "unpaid_family")
    log_wage[emask] = (
        1.8
        + 0.08 * educ_years[emask]  # ~8% return to education
        + 0.04 * experience[emask]
        - 0.0006 * experience[emask] ** 2
        - 0.18 * female[emask]  # gender gap
        + 0.25 * urban[emask].astype(float)
        + 0.35 * formal[emask]  # formality premium
        + rng.normal(0, 0.4, emask.sum())
    )

    # Monthly wage (USD PPP)
    monthly_wage = np.full(n, np.nan)
    monthly_wage[emask] = np.round(np.exp(log_wage[emask]), 2)

    # Hours worked per week
    hours_weekly = np.full(n, np.nan)
    hours_weekly[employed_mask] = np.clip(
        rng.normal(42, 12, employed_mask.sum())
        + 5 * formal[employed_mask]
        - 3 * female[employed_mask],
        5, 80
    ).round(0)

    # Underemployment (<35 hours but wants more)
    underemployed = np.zeros(n, dtype=int)
    short_hours = employed_mask & (hours_weekly < 35)
    underemployed[short_hours] = rng.binomial(1, 0.6, short_hours.sum())

    # Written contract (formal sector mostly)
    has_contract = np.zeros(n, dtype=int)
    has_contract[employed_mask] = rng.binomial(
        1, np.clip(0.1 + 0.6 * formal[employed_mask], 0, 1)
    )

    # Social protection
    has_pension = formal * rng.binomial(1, 0.55, n) * (emp_status != "unemployed").astype(int)
    has_health_insurance = formal * rng.binomial(1, 0.45, n) * (emp_status != "unemployed").astype(int)

    # Job search (unemployed)
    actively_searching = np.zeros(n, dtype=int)
    unemp_mask = emp_status == "unemployed"
    actively_searching[unemp_mask] = rng.binomial(1, 0.65, unemp_mask.sum())
    months_unemployed = np.full(n, np.nan)
    months_unemployed[unemp_mask] = np.clip(rng.exponential(6, unemp_mask.sum()), 0.5, 48).round(1)

    # Remittance amount
    remittance_usd = np.full(n, 0.0)
    rem_mask = sends_remittances.astype(bool)
    remittance_usd[rem_mask] = np.round(rng.lognormal(3.5, 0.8, rem_mask.sum()), 2)

    # Survey quarter (seasonality)
    quarter = rng.choice([1, 2, 3, 4], n)

    df = pd.DataFrame({
        "individual_id": ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "education_years": educ_years,
        "experience_years": experience,
        "married": married,
        "migrant": migrant,
        "employment_status": emp_status,
        "sector": sector,
        "formal_sector": formal,
        "monthly_wage_usd": monthly_wage,
        "hours_per_week": hours_weekly,
        "underemployed": underemployed,
        "has_written_contract": has_contract,
        "has_pension": has_pension,
        "has_health_insurance": has_health_insurance,
        "actively_searching": actively_searching,
        "months_unemployed": months_unemployed,
        "sends_remittances": sends_remittances,
        "monthly_remittance_usd": remittance_usd,
        "survey_quarter": quarter,
    })

    df = inject_missing(df,
        columns=["monthly_wage_usd", "hours_per_week", "education_years", "months_unemployed"],
        rates=[0.08, 0.04, 0.02, 0.06],
        rng=rng, mechanism="MNAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + slope * np.asarray(z))


def _sector_probs(is_urban):
    if is_urban:
        return [0.05, 0.12, 0.10, 0.25, 0.10, 0.22, 0.10, 0.06]
    else:
        return [0.45, 0.05, 0.08, 0.15, 0.05, 0.10, 0.05, 0.07]
