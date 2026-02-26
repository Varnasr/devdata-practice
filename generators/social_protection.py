"""
Generator: Social Protection & Cash Transfers
──────────────────────────────────────────────
Simulates a social protection programme dataset with beneficiary registry,
cash transfer disbursement, conditionality compliance, graduation criteria,
and wellbeing outcomes.

Rows: ~20k beneficiary households.

Realistic features:
  • Multiple transfer modalities (mobile money, cash-in-hand, voucher)
  • Conditionalities (health visits, school attendance) with compliance
  • Graduation model with thresholds
  • Consumption smoothing effects
  • Productive asset accumulation
  • Dependency vs. graduation debate metrics
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES, random_dates


def generate(n_households: int = 20000, seed: int = 710) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    hh_ids = household_ids(rng, n, prefix="BEN")
    districts, urban = pick_districts(rng, n, urban_share=0.25)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    hh_size = rng.choice(range(1, 11), n,
                         p=[0.02, 0.05, 0.09, 0.14, 0.20, 0.20, 0.14, 0.08, 0.05, 0.03])
    n_children = np.clip(rng.poisson(hh_size * 0.4), 0, hh_size - 1)
    head_female = rng.binomial(1, 0.42, n)  # higher FHH in SP programmes
    head_age = rng.integers(18, 75, n)
    head_educ = np.clip(rng.normal(4.5, 3, n), 0, 16).astype(int)
    head_disabled = rng.binomial(1, 0.08, n)
    head_elderly = (head_age >= 60).astype(int)

    # Poverty status (pre-programme)
    wealth_baseline = rng.normal(-0.5, 0.8, n)  # SP targets poor
    baseline_consumption = np.round(np.exp(3.5 + 0.4 * wealth_baseline + rng.normal(0, 0.3, n)), 2)
    poverty_line = 65.0
    baseline_poor = (baseline_consumption < poverty_line).astype(int)

    # Programme details
    programme_type = rng.choice(
        ["unconditional_cash", "conditional_cash", "public_works", "cash_plus",
         "food_voucher", "school_feeding"],
        n, p=[0.25, 0.22, 0.15, 0.12, 0.14, 0.12]
    )
    transfer_modality = rng.choice(
        ["mobile_money", "cash_in_hand", "bank_transfer", "voucher", "in_kind"],
        n, p=[0.35, 0.25, 0.15, 0.15, 0.10]
    )
    enrollment_date = random_dates(rng, n, "2019-01-01", "2023-06-30")
    months_enrolled = np.clip(rng.poisson(18, n), 3, 60)

    # Transfer amount (monthly USD)
    base_transfer = np.where(
        programme_type == "unconditional_cash", rng.normal(25, 5, n),
        np.where(programme_type == "conditional_cash", rng.normal(30, 6, n),
        np.where(programme_type == "public_works", rng.normal(35, 8, n),
        np.where(programme_type == "cash_plus", rng.normal(40, 8, n),
        np.where(programme_type == "food_voucher", rng.normal(20, 4, n),
                 rng.normal(15, 3, n))))))
    monthly_transfer_usd = np.round(np.clip(base_transfer, 5, 80), 2)
    total_received_usd = np.round(monthly_transfer_usd * months_enrolled, 2)

    # Conditionality compliance
    has_conditionality = np.isin(programme_type, ["conditional_cash", "cash_plus", "school_feeding"])
    cond_health_visits = np.where(has_conditionality, rng.binomial(1, 0.72 + 0.05 * head_educ / 18), 0)
    cond_school_attendance = np.where(has_conditionality & (n_children > 0),
                                      rng.binomial(1, 0.78), 0)
    compliant = np.where(has_conditionality,
                         cond_health_visits | cond_school_attendance, 1)

    # Payment regularity
    pct_payments_received = np.clip(rng.beta(8, 1.5, n) * 100, 40, 100).round(1)
    delayed_payment = rng.binomial(1, 0.20, n)

    # --- Outcomes (endline) ---
    treatment_effect = monthly_transfer_usd / baseline_consumption * 0.8
    endline_consumption = np.round(
        baseline_consumption * (1 + treatment_effect + 0.03 * months_enrolled / 12
                                + rng.normal(0, 0.1, n)),
        2
    )
    endline_poor = (endline_consumption < poverty_line).astype(int)

    # Food consumption score (0-112)
    fcs_baseline = np.clip(rng.normal(35 + 5 * wealth_baseline, 10, n), 0, 112).round(1)
    fcs_endline = np.clip(fcs_baseline + 8 * (monthly_transfer_usd / 30) + rng.normal(0, 5, n), 0, 112).round(1)

    # Coping strategies (rCSI 0-56, lower is better)
    rcsi_baseline = np.clip(rng.normal(25 - 3 * wealth_baseline, 8, n), 0, 56).round(0).astype(int)
    rcsi_endline = np.clip(rcsi_baseline - 5 * (monthly_transfer_usd / 30) + rng.normal(0, 4, n), 0, 56).round(0).astype(int)

    # Savings
    has_savings_baseline = rng.binomial(1, _logistic(0.15, wealth_baseline, 0.4))
    has_savings_endline = rng.binomial(1, _logistic(0.25, wealth_baseline + 0.3 * months_enrolled / 24, 0.4))
    savings_amount_usd = np.where(has_savings_endline,
        np.round(rng.lognormal(2.5 + 0.2 * months_enrolled / 12, 0.6, n), 2), 0)

    # Productive assets
    n_productive_assets_baseline = np.clip(rng.poisson(np.clip(1 + wealth_baseline, 0.1, 5)), 0, 8)
    n_productive_assets_endline = np.clip(
        n_productive_assets_baseline + rng.poisson(np.clip(0.5 * months_enrolled / 12, 0.1, 3)),
        0, 10)

    # Graduation criteria
    meets_food_security = (fcs_endline >= 42).astype(int)
    meets_asset_threshold = (n_productive_assets_endline >= 3).astype(int)
    meets_savings_threshold = (savings_amount_usd >= 50).astype(int)
    graduation_score = meets_food_security + meets_asset_threshold + meets_savings_threshold
    graduated = (graduation_score >= 2).astype(int)

    # Dependency indicator (subjective)
    would_cope_without_transfer = rng.binomial(1, _logistic(0.30, 0.1 * graduation_score + 0.1 * wealth_baseline, 0.4))

    df = pd.DataFrame({
        "household_id": hh_ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "household_size": hh_size,
        "n_children": n_children,
        "head_female": head_female, "head_age": head_age,
        "head_education_years": head_educ,
        "head_disabled": head_disabled, "head_elderly": head_elderly,
        "programme_type": programme_type, "transfer_modality": transfer_modality,
        "enrollment_date": enrollment_date, "months_enrolled": months_enrolled,
        "monthly_transfer_usd": monthly_transfer_usd,
        "total_received_usd": total_received_usd,
        "has_conditionality": has_conditionality.astype(int),
        "conditionality_compliant": compliant,
        "pct_payments_received": pct_payments_received,
        "delayed_payment": delayed_payment,
        "baseline_consumption_usd": baseline_consumption,
        "endline_consumption_usd": endline_consumption,
        "baseline_poor": baseline_poor, "endline_poor": endline_poor,
        "fcs_baseline": fcs_baseline, "fcs_endline": fcs_endline,
        "rcsi_baseline": rcsi_baseline, "rcsi_endline": rcsi_endline,
        "has_savings_baseline": has_savings_baseline,
        "has_savings_endline": has_savings_endline,
        "savings_amount_usd": savings_amount_usd,
        "productive_assets_baseline": n_productive_assets_baseline,
        "productive_assets_endline": n_productive_assets_endline,
        "meets_food_security": meets_food_security,
        "meets_asset_threshold": meets_asset_threshold,
        "meets_savings_threshold": meets_savings_threshold,
        "graduation_score": graduation_score,
        "graduated": graduated,
        "would_cope_without_transfer": would_cope_without_transfer,
    })

    df = inject_missing(df,
        columns=["endline_consumption_usd", "fcs_endline", "savings_amount_usd", "rcsi_endline"],
        rates=[0.05, 0.04, 0.06, 0.04],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
