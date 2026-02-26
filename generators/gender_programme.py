"""
Generator: Gender Programme
───────────────────────────
Simulates a gender-focused programme dataset with GBV prevalence,
women's empowerment indices (WEAI-like), decision-making power,
time use, access to resources, and programme intervention data.

Rows: ~25k women/girls (individual-level).

Realistic features:
  • Women's Empowerment in Agriculture Index (WEAI) domains
  • Intra-household bargaining / decision-making indicators
  • GBV prevalence (physical, emotional, economic) with underreporting
  • Time-use diary: care work, productive work, leisure
  • SRH indicators: contraceptive use, unmet need
  • Programme participation and empowerment change over time
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, household_ids


def generate(n_individuals: int = 25000, seed: int = 701) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    ids = [f"GEN-{i:06d}" for i in range(1, n + 1)]
    hh_ids = household_ids(rng, n)
    districts, urban = pick_districts(rng, n, urban_share=0.32)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    age = rng.integers(15, 55, n)
    married = rng.binomial(1, np.where(age < 18, 0.08, np.where(age < 25, 0.45, 0.75)))
    educ_years = np.clip(rng.normal(5 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    n_children = np.clip(rng.poisson(np.where(age < 20, 0.5, np.where(age < 30, 1.8, 3.2))), 0, 10)
    hh_size = np.clip(n_children + rng.integers(1, 4, n), 1, 12)

    # Wealth proxy
    wealth = rng.normal(0, 1, n) + 0.3 * educ_years / 10 + 0.3 * urban.astype(float)

    # Programme participation
    programme_participant = rng.binomial(1, 0.40, n)
    programme_type = np.where(
        programme_participant,
        rng.choice(["cash_transfer", "skills_training", "savings_group", "awareness_campaign", "legal_aid"],
                   n, p=[0.25, 0.22, 0.20, 0.18, 0.15]),
        "none"
    )

    # --- Decision-making (5 domains) ---
    _emp = lambda base: np.clip(base + 0.1 * educ_years + 0.15 * programme_participant + rng.normal(0, 0.3, n), 0, 1)
    decides_own_healthcare = rng.binomial(1, _emp(0.35))
    decides_large_purchases = rng.binomial(1, _emp(0.25))
    decides_visits_family = rng.binomial(1, _emp(0.40))
    decides_children_education = rng.binomial(1, _emp(0.45))
    decides_own_earnings = rng.binomial(1, _emp(0.50))
    decision_making_score = (decides_own_healthcare + decides_large_purchases +
                             decides_visits_family + decides_children_education +
                             decides_own_earnings)

    # --- Economic empowerment ---
    owns_land = rng.binomial(1, _logistic(0.15, wealth, 0.4))
    owns_house = rng.binomial(1, _logistic(0.12, wealth, 0.4))
    has_bank_account = rng.binomial(1, _logistic(0.30, wealth + 0.2 * programme_participant, 0.5))
    has_mobile_money = rng.binomial(1, _logistic(0.40, wealth + 0.15 * programme_participant, 0.4))
    earns_own_income = rng.binomial(1, _logistic(0.45, wealth + 0.1 * educ_years / 10, 0.4))
    monthly_income_usd = np.where(
        earns_own_income,
        np.round(np.exp(rng.normal(3.5 + 0.3 * wealth, 0.5, n)), 2),
        0
    )

    # --- Time use (hours per day) ---
    care_work_hours = np.clip(rng.normal(5.5 - 0.5 * programme_participant - 0.3 * wealth, 1.5, n), 0, 14)
    productive_work_hours = np.clip(rng.normal(3 + 0.5 * earns_own_income, 1.5, n), 0, 12)
    leisure_hours = np.clip(24 - care_work_hours - productive_work_hours - 8 + rng.normal(0, 0.5, n), 0, 8)

    # --- GBV indicators (sensitive — with underreporting) ---
    # True prevalence is higher than reported
    true_physical_gbv = rng.binomial(1, _logistic(0.25, -wealth - 0.05 * educ_years, 0.3))
    true_emotional_gbv = rng.binomial(1, _logistic(0.35, -wealth - 0.03 * educ_years, 0.25))
    true_economic_gbv = rng.binomial(1, _logistic(0.30, -wealth, 0.3))
    # Underreporting: only 40-60% of true cases are disclosed
    reporting_rate = 0.40 + 0.10 * programme_participant + 0.05 * educ_years / 18
    reported_physical_gbv = true_physical_gbv * rng.binomial(1, np.clip(reporting_rate, 0.2, 0.8))
    reported_emotional_gbv = true_emotional_gbv * rng.binomial(1, np.clip(reporting_rate, 0.2, 0.8))
    reported_economic_gbv = true_economic_gbv * rng.binomial(1, np.clip(reporting_rate + 0.05, 0.2, 0.8))

    # Sought help
    sought_help = ((reported_physical_gbv | reported_emotional_gbv) *
                   rng.binomial(1, _logistic(0.30, wealth + 0.2 * programme_participant, 0.4)))

    # --- SRH ---
    using_contraception = np.where(
        married & (age >= 15) & (age <= 49),
        rng.binomial(1, _logistic(0.35, wealth + 0.1 * educ_years / 10 + 0.1 * programme_participant, 0.4)),
        0
    )
    unmet_need_fp = np.where(
        married & (age >= 15) & (age <= 49) & ~using_contraception.astype(bool),
        rng.binomial(1, 0.22), 0
    )

    # --- Empowerment composite (WEAI-like, 0-1) ---
    empowerment_index = np.clip(
        (decision_making_score / 5) * 0.3
        + has_bank_account * 0.15
        + earns_own_income * 0.15
        + owns_land * 0.10
        + (1 - care_work_hours / 14) * 0.10
        + using_contraception * 0.10
        + (educ_years / 18) * 0.10
        + rng.normal(0, 0.05, n),
        0, 1
    )

    df = pd.DataFrame({
        "individual_id": ids, "household_id": hh_ids,
        "country": countries, "district": districts, "urban": urban.astype(int),
        "age": age, "married": married, "education_years": educ_years,
        "n_children": n_children, "household_size": hh_size,
        "programme_participant": programme_participant,
        "programme_type": programme_type,
        "decides_own_healthcare": decides_own_healthcare,
        "decides_large_purchases": decides_large_purchases,
        "decides_visits_family": decides_visits_family,
        "decides_children_education": decides_children_education,
        "decides_own_earnings": decides_own_earnings,
        "decision_making_score": decision_making_score,
        "owns_land": owns_land, "owns_house": owns_house,
        "has_bank_account": has_bank_account, "has_mobile_money": has_mobile_money,
        "earns_own_income": earns_own_income,
        "monthly_income_usd": monthly_income_usd,
        "care_work_hours_day": np.round(care_work_hours, 1),
        "productive_work_hours_day": np.round(productive_work_hours, 1),
        "leisure_hours_day": np.round(leisure_hours, 1),
        "reported_physical_gbv": reported_physical_gbv,
        "reported_emotional_gbv": reported_emotional_gbv,
        "reported_economic_gbv": reported_economic_gbv,
        "sought_help_gbv": sought_help,
        "using_modern_contraception": using_contraception,
        "unmet_need_family_planning": unmet_need_fp,
        "empowerment_index": np.round(empowerment_index, 3),
    })

    df = inject_missing(df,
        columns=["monthly_income_usd", "reported_physical_gbv", "care_work_hours_day", "education_years"],
        rates=[0.06, 0.12, 0.04, 0.03],
        rng=rng, mechanism="MNAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + slope * np.asarray(z))
