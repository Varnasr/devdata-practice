"""
Generator: ILO Decent Work Indicators
──────────────────────────────────────
Simulates worker-level data following the ILO decent work framework covering
employment status, earnings, social protection, working conditions, freedom of
association, and work-life balance.

Rows: ~25k workers.

Realistic features:
  • Formal/informal sector with associated protections
  • Gender pay gap embedded in earnings
  • Occupational segregation by gender
  • Social protection linked to formality
  • Working conditions and safety indicators
  • Union membership and collective bargaining
  • MAR missingness mechanism
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_workers: int = 25000, seed: int = 803) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_workers

    # --- Demographics ---
    ids = [f"WKR-{i:07d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.40)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.48, n)
    age = rng.integers(15, 65, n)
    educ_years = np.clip(
        rng.normal(7 + 2.5 * urban.astype(float), 3.5, n), 0, 20
    ).astype(int)

    # Wealth latent variable
    wealth = rng.normal(0, 1, n) + 0.3 * urban.astype(float) + 0.06 * educ_years

    # --- Sector assignment with occupational segregation ---
    all_sectors = ["agriculture", "manufacturing", "services", "construction",
                   "mining", "domestic_work", "transport", "trade"]

    def _sector_probs(is_urban, is_female):
        if is_urban and is_female:
            return [0.04, 0.10, 0.30, 0.02, 0.01, 0.22, 0.03, 0.28]
        elif is_urban and not is_female:
            return [0.05, 0.15, 0.25, 0.15, 0.04, 0.02, 0.12, 0.22]
        elif not is_urban and is_female:
            return [0.45, 0.05, 0.12, 0.01, 0.01, 0.15, 0.01, 0.20]
        else:
            return [0.40, 0.08, 0.10, 0.12, 0.06, 0.02, 0.08, 0.14]

    sector = np.array([
        rng.choice(all_sectors, p=_sector_probs(urban[i], female[i]))
        for i in range(n)
    ])

    # --- Employment status ---
    emp_statuses = ["formal_wage", "informal_wage", "self_employed", "casual_daily", "unpaid_family"]
    emp_probs = np.column_stack([
        _logistic(0.20, (educ_years - 8) / 4 + 0.3 * urban.astype(float), 0.5),
        _logistic(0.25, -0.1 * educ_years / 10, 0.3),
        np.full(n, 0.25),
        _logistic(0.15, -0.1 * urban.astype(float) - educ_years / 20, 0.3),
        _logistic(0.08, 0.15 * female, 0.3),
    ])
    emp_probs = emp_probs / emp_probs.sum(axis=1, keepdims=True)
    employment_status = np.array([
        rng.choice(emp_statuses, p=emp_probs[i]) for i in range(n)
    ])

    is_formal = (employment_status == "formal_wage").astype(float)

    # --- Hours worked ---
    hours_worked_weekly = np.clip(
        rng.normal(44, 12, n) + 5 * is_formal - 3 * female, 5, 80
    ).round(0).astype(int)
    underemployed = (hours_worked_weekly < 35).astype(int) * rng.binomial(1, 0.55, n)

    # --- Earnings (with gender pay gap) ---
    experience = np.clip(age - educ_years - 6, 0, 45)
    log_earnings = (
        3.5
        + 0.07 * educ_years
        + 0.03 * experience
        - 0.0005 * experience ** 2
        - 0.20 * female  # gender pay gap ~20%
        + 0.30 * urban.astype(float)
        + 0.40 * is_formal
        + rng.normal(0, 0.45, n)
    )

    monthly_earnings_usd = np.where(
        employment_status == "unpaid_family",
        0.0,
        np.round(np.exp(log_earnings), 2),
    )
    hourly_wage_usd = np.where(
        (hours_worked_weekly > 0) & (employment_status != "unpaid_family"),
        np.round(monthly_earnings_usd / np.maximum(hours_worked_weekly * 4.33, 1), 2),
        0.0,
    )

    below_minimum_wage = rng.binomial(
        1, _logistic(0.30, -is_formal * 1.5 - educ_years / 15, 0.4)
    )
    paid_regularly = rng.binomial(
        1, _logistic(0.55, is_formal * 1.2 + 0.1 * urban.astype(float), 0.4)
    )
    earnings_in_kind = rng.binomial(
        1, _logistic(0.20, -is_formal * 0.8 - 0.1 * urban.astype(float), 0.3)
    )

    # --- Social protection (strongly linked to formality) ---
    has_written_contract = rng.binomial(
        1, np.clip(0.08 + 0.70 * is_formal + 0.05 * urban.astype(float), 0, 1)
    )
    has_social_security = rng.binomial(
        1, np.clip(0.05 + 0.55 * is_formal, 0, 1)
    )
    has_health_insurance = rng.binomial(
        1, np.clip(0.08 + 0.50 * is_formal + 0.05 * wealth / 3, 0, 1)
    )
    has_pension = rng.binomial(
        1, np.clip(0.04 + 0.45 * is_formal, 0, 1)
    )
    paid_leave_days = np.where(
        is_formal > 0.5,
        np.clip(rng.normal(15, 5, n), 0, 30).astype(int),
        np.where(rng.random(n) < 0.10, rng.integers(1, 10, n), 0),
    )
    maternity_leave_available = rng.binomial(
        1, np.clip(0.05 + 0.55 * is_formal, 0, 1)
    )

    # --- Working conditions ---
    occupational_safety_training = rng.binomial(
        1, _logistic(0.20, is_formal * 0.8 + 0.1 * urban.astype(float), 0.3)
    )
    experienced_injury_12m = rng.binomial(
        1, _logistic(0.08, -occupational_safety_training * 0.3
                     + 0.15 * (sector == "construction").astype(float)
                     + 0.10 * (sector == "mining").astype(float), 0.3)
    )
    workplace_harassment = rng.binomial(
        1, _logistic(0.12, 0.08 * female - 0.1 * is_formal, 0.3)
    )
    child_labor_hh = rng.binomial(
        1, _logistic(0.08, -educ_years / 15 - 0.1 * wealth - 0.1 * urban.astype(float), 0.3)
    )

    # --- Freedom of association ---
    freedom_of_association = rng.binomial(
        1, _logistic(0.55, is_formal * 0.4 + 0.1 * urban.astype(float), 0.3)
    )
    member_of_union = rng.binomial(
        1, _logistic(0.10, is_formal * 0.8 + 0.05 * educ_years / 10, 0.4)
    )
    collective_bargaining_covered = rng.binomial(
        1, np.clip(0.05 + 0.35 * member_of_union + 0.15 * is_formal, 0, 1)
    )

    # --- Work-life ---
    commute_time_min = np.clip(
        rng.lognormal(3.0 + 0.3 * urban.astype(float), 0.6, n), 5, 180
    ).astype(int)
    satisfied_with_job = np.clip(
        rng.normal(3.0 + 0.3 * is_formal + 0.1 * wealth - 0.1 * hours_worked_weekly / 50, 0.9, n),
        1, 5
    ).round(1)
    would_change_job = rng.binomial(
        1, _logistic(0.35, -satisfied_with_job / 5, 0.5)
    )

    # --- Informal economy indicators ---
    operates_without_registration = rng.binomial(
        1, _logistic(0.40, -is_formal * 1.5 - educ_years / 15, 0.4)
    )
    no_bookkeeping = rng.binomial(
        1, _logistic(0.50, -is_formal * 1.2 - educ_years / 18, 0.3)
    )
    works_from_home = rng.binomial(
        1, _logistic(0.15, -is_formal * 0.5 + 0.1 * female
                     + 0.2 * (sector == "domestic_work").astype(float), 0.3)
    )

    df = pd.DataFrame({
        "worker_id": ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "education_years": educ_years,
        "sector": sector,
        "employment_status": employment_status,
        "hours_worked_weekly": hours_worked_weekly,
        "underemployed": underemployed,
        "monthly_earnings_usd": monthly_earnings_usd,
        "hourly_wage_usd": hourly_wage_usd,
        "below_minimum_wage": below_minimum_wage,
        "paid_regularly": paid_regularly,
        "earnings_in_kind": earnings_in_kind,
        "has_written_contract": has_written_contract,
        "has_social_security": has_social_security,
        "has_health_insurance": has_health_insurance,
        "has_pension": has_pension,
        "paid_leave_days": paid_leave_days,
        "maternity_leave_available": maternity_leave_available,
        "occupational_safety_training": occupational_safety_training,
        "experienced_injury_12m": experienced_injury_12m,
        "workplace_harassment": workplace_harassment,
        "child_labor_hh": child_labor_hh,
        "freedom_of_association": freedom_of_association,
        "member_of_union": member_of_union,
        "collective_bargaining_covered": collective_bargaining_covered,
        "commute_time_min": commute_time_min,
        "satisfied_with_job": satisfied_with_job,
        "would_change_job": would_change_job,
        "operates_without_registration": operates_without_registration,
        "no_bookkeeping": no_bookkeeping,
        "works_from_home": works_from_home,
    })

    df = inject_missing(df,
        columns=["monthly_earnings_usd", "hourly_wage_usd", "hours_worked_weekly",
                 "satisfied_with_job", "paid_leave_days"],
        rates=[0.07, 0.07, 0.04, 0.05, 0.06],
        rng=rng, mechanism="MAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + slope * np.asarray(z))
