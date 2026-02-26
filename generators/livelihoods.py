"""
Generator: Livelihoods & Economic Strengthening
─────────────────────────────────────────────────
Simulates household-level livelihoods and economic-strengthening programme data,
covering income diversification, savings groups (VSLA/SILC), vocational
training, asset accumulation, food security indicators, market linkages,
youth employment, and financial inclusion.

Rows: one per household (~20k), with baseline and endline measurements for
asset and food-security variables.

Realistic features:
  • Income diversification index (Shannon-like)
  • VSLA/SILC membership with savings, share-outs, and borrowing
  • Vocational training and apprenticeship completion rates
  • Asset index growth between baseline and endline
  • Food Consumption Score (FCS, 0-112) and reduced Coping Strategies
    Index (rCSI, 0-56) correlated with wealth
  • Market linkage and business skills score
  • Youth-specific (15-35) employment indicators
  • Financial inclusion: mobile money, formal banking, credit access
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_households: int = 20000, seed: int = 602) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    # --- IDs & geography ---
    hh_ids = household_ids(rng, n)
    districts, urban = pick_districts(rng, n, urban_share=0.32)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # --- Household demographics ---
    hh_size = rng.choice(range(1, 11), n,
                         p=[0.02, 0.05, 0.10, 0.15, 0.22, 0.18, 0.13, 0.08, 0.05, 0.02])
    head_female = rng.binomial(1, 0.30, n)
    head_age = rng.integers(20, 70, n)
    head_educ = np.clip(rng.normal(5.5 + 1.5 * urban.astype(float), 3.5, n), 0, 18).astype(int)

    # Latent wealth (drives many outcomes)
    wealth_latent = rng.normal(0, 1, n) + 0.4 * urban.astype(float) + 0.15 * head_educ / 10
    wealth_quintile = pd.qcut(wealth_latent, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Programme treatment arm (for evaluation structure)
    treatment = rng.choice(["control", "livelihoods_only", "livelihoods_plus_savings"],
                           n, p=[0.35, 0.35, 0.30])

    # ------------------------------------------------------------------ #
    # Income diversification
    # ------------------------------------------------------------------ #
    income_sources = rng.choice(
        ["crop_farming", "livestock", "wage_employment", "casual_labor",
         "small_business", "remittances", "fishing", "artisan_craft"],
        size=(n, 3),  # up to 3 sources per household
    )
    n_income_sources = np.clip(rng.poisson(np.clip(2.2 + 0.3 * wealth_latent, 0.1, 20)), 1, 7)
    primary_income = rng.choice(
        ["crop_farming", "livestock", "wage_employment", "casual_labor",
         "small_business", "remittances", "fishing", "artisan_craft"],
        n, p=[0.22, 0.10, 0.15, 0.18, 0.15, 0.06, 0.04, 0.10],
    )
    monthly_income_usd = np.round(
        np.exp(3.8 + 0.4 * wealth_latent + 0.15 * n_income_sources / 5 + rng.normal(0, 0.4, n)), 2
    )
    income_diversification_idx = np.round(np.clip(
        0.15 * n_income_sources + rng.normal(0, 0.08, n), 0, 1
    ), 3)

    # ------------------------------------------------------------------ #
    # Enterprise development
    # ------------------------------------------------------------------ #
    owns_enterprise = rng.binomial(1, _logistic(0.30, wealth_latent * 0.3 + 0.1 * head_educ / 10, 0.4))
    enterprise_type = np.where(
        owns_enterprise,
        rng.choice(["retail_shop", "food_vending", "tailoring", "agro_processing",
                     "transport", "construction", "services_other"],
                   n, p=[0.25, 0.20, 0.12, 0.15, 0.08, 0.08, 0.12]),
        "none",
    )
    enterprise_monthly_revenue_usd = np.where(
        owns_enterprise,
        np.round(np.exp(4.0 + 0.35 * wealth_latent + rng.normal(0, 0.5, n)), 2),
        0.0,
    )
    enterprise_employees = np.where(
        owns_enterprise,
        np.clip(rng.poisson(np.clip(1.5 + 0.5 * wealth_latent, 0.1, 20)), 0, 15),
        0,
    )

    # ------------------------------------------------------------------ #
    # Savings groups (VSLA / SILC)
    # ------------------------------------------------------------------ #
    in_treatment = (treatment != "control").astype(float)
    vsla_member = rng.binomial(1, _logistic(
        0.25, 0.4 * in_treatment + 0.1 * wealth_latent - 0.15 * urban.astype(float), 0.4
    ))
    vsla_savings_usd = np.where(
        vsla_member,
        np.round(rng.exponential(25 + 10 * wealth_latent.clip(-2, 3)), 2),
        0.0,
    )
    vsla_shareout_usd = np.where(
        vsla_member,
        np.round(vsla_savings_usd * rng.uniform(1.05, 1.35, n), 2),
        0.0,
    )
    vsla_borrowed = np.where(
        vsla_member, rng.binomial(1, 0.55, n), 0
    )
    vsla_loan_usd = np.where(
        vsla_borrowed,
        np.round(rng.exponential(30 + 8 * wealth_latent.clip(-2, 3)), 2),
        0.0,
    )

    # ------------------------------------------------------------------ #
    # Vocational training & employment
    # ------------------------------------------------------------------ #
    received_vocational_training = rng.binomial(1, _logistic(
        0.15, 0.3 * in_treatment + 0.1 * head_educ / 10, 0.3
    ))
    training_type = np.where(
        received_vocational_training,
        rng.choice(["carpentry", "welding", "tailoring", "mechanics",
                     "ICT", "agriculture", "hairdressing", "masonry"],
                   n, p=[0.12, 0.10, 0.15, 0.10, 0.13, 0.18, 0.12, 0.10]),
        "none",
    )
    completed_apprenticeship = np.where(
        received_vocational_training,
        rng.binomial(1, _logistic(0.60, 0.1 * wealth_latent, 0.3)), 0
    )
    in_wage_employment = rng.binomial(1, _logistic(
        0.20, 0.2 * wealth_latent + 0.15 * urban.astype(float) + 0.1 * head_educ / 10, 0.4
    ))

    # ------------------------------------------------------------------ #
    # Asset accumulation (baseline → endline)
    # ------------------------------------------------------------------ #
    asset_names = ["radio", "mobile_phone", "bicycle", "solar_panel",
                   "improved_stove", "iron_roof", "livestock_tlu"]
    asset_probs_base = np.array([0.55, 0.70, 0.35, 0.15, 0.20, 0.40, 0.30])

    baseline_assets = {}
    endline_assets = {}
    for i, name in enumerate(asset_names):
        p_bl = _logistic(asset_probs_base[i], wealth_latent * 0.3, slope=0.6)
        bl = rng.binomial(1, p_bl)
        baseline_assets[f"baseline_{name}"] = bl
        # Endline: modest improvement, larger in treatment group
        p_gain = _logistic(0.10, 0.15 * in_treatment + 0.05 * wealth_latent, 0.3)
        gained = rng.binomial(1, p_gain)
        endline_assets[f"endline_{name}"] = np.maximum(bl, gained)

    baseline_asset_index = np.round(
        sum(baseline_assets.values()) / len(asset_names), 3
    )
    endline_asset_index = np.round(
        sum(endline_assets.values()) / len(asset_names), 3
    )

    # ------------------------------------------------------------------ #
    # Food security
    # ------------------------------------------------------------------ #
    # Food Consumption Score (0-112, weighted frequency of food groups)
    fcs = np.clip(
        rng.normal(48 + 8 * wealth_latent + 4 * in_treatment + 3 * urban.astype(float), 14, n),
        0, 112,
    ).round(1)
    fcs_category = np.where(fcs <= 21, "poor",
                   np.where(fcs <= 35, "borderline", "acceptable"))

    # Reduced Coping Strategies Index (rCSI, 0-56, lower is better)
    rcsi = np.clip(
        rng.normal(18 - 4 * wealth_latent - 2 * in_treatment, 8, n),
        0, 56,
    ).round(1)

    # ------------------------------------------------------------------ #
    # Market linkage & business skills
    # ------------------------------------------------------------------ #
    market_linkage = rng.binomial(1, _logistic(
        0.20, 0.15 * wealth_latent + 0.1 * urban.astype(float) + 0.15 * owns_enterprise, 0.4
    ))
    business_skills_score = np.clip(
        rng.normal(45 + 8 * head_educ / 10 + 5 * received_vocational_training
                   + 3 * in_treatment + 4 * wealth_latent, 12, n),
        0, 100,
    ).round(1)

    # ------------------------------------------------------------------ #
    # Youth-specific indicators (head age 15-35)
    # ------------------------------------------------------------------ #
    is_youth = ((head_age >= 15) & (head_age <= 35)).astype(float)
    youth_employed = np.where(
        is_youth, rng.binomial(1, _logistic(0.35, 0.2 * wealth_latent + 0.1 * head_educ / 10, 0.3)), -1
    )
    youth_neet = np.where(
        is_youth,
        rng.binomial(1, _logistic(0.25, -0.2 * wealth_latent - 0.1 * head_educ / 10, 0.3)),
        -1,
    )
    youth_in_training = np.where(
        is_youth, received_vocational_training * is_youth.astype(int), -1
    )

    # ------------------------------------------------------------------ #
    # Financial inclusion
    # ------------------------------------------------------------------ #
    has_mobile_money = rng.binomial(1, _logistic(
        0.45, 0.2 * wealth_latent + 0.25 * urban.astype(float), 0.4
    ))
    has_bank_account = rng.binomial(1, _logistic(
        0.15, 0.3 * wealth_latent + 0.2 * urban.astype(float) + 0.1 * head_educ / 10, 0.5
    ))
    accessed_credit_12m = rng.binomial(1, _logistic(
        0.18, 0.15 * wealth_latent + 0.1 * vsla_member + 0.1 * has_bank_account, 0.4
    ))
    credit_source = np.where(
        accessed_credit_12m,
        rng.choice(["vsla_silc", "mfi", "bank", "mobile_lender", "informal_moneylender", "family"],
                   n, p=[0.25, 0.20, 0.12, 0.15, 0.13, 0.15]),
        "none",
    )
    total_savings_usd = np.round(
        np.exp(2.5 + 0.5 * wealth_latent + 0.3 * has_bank_account
               + 0.2 * has_mobile_money + rng.normal(0, 0.5, n)),
        2,
    )

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "household_id": hh_ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "treatment_arm": treatment,
        "household_size": hh_size,
        "head_female": head_female,
        "head_age": head_age,
        "head_education_years": head_educ,
        "wealth_quintile": wealth_quintile,
        # Income
        "primary_income_source": primary_income,
        "n_income_sources": n_income_sources,
        "monthly_income_usd": monthly_income_usd,
        "income_diversification_index": income_diversification_idx,
        # Enterprise
        "owns_enterprise": owns_enterprise,
        "enterprise_type": enterprise_type,
        "enterprise_monthly_revenue_usd": enterprise_monthly_revenue_usd,
        "enterprise_employees": enterprise_employees,
        # Savings groups
        "vsla_member": vsla_member,
        "vsla_savings_usd": vsla_savings_usd,
        "vsla_shareout_usd": vsla_shareout_usd,
        "vsla_borrowed": vsla_borrowed,
        "vsla_loan_usd": vsla_loan_usd,
        # Training & employment
        "received_vocational_training": received_vocational_training,
        "training_type": training_type,
        "completed_apprenticeship": completed_apprenticeship,
        "in_wage_employment": in_wage_employment,
        # Assets (baseline & endline)
        **baseline_assets,
        **endline_assets,
        "baseline_asset_index": baseline_asset_index,
        "endline_asset_index": endline_asset_index,
        # Food security
        "food_consumption_score": fcs,
        "fcs_category": fcs_category,
        "reduced_coping_strategies_index": rcsi,
        # Market & skills
        "market_linkage": market_linkage,
        "business_skills_score": business_skills_score,
        # Youth
        "is_youth_head": is_youth.astype(int),
        "youth_employed": youth_employed,
        "youth_neet": youth_neet,
        "youth_in_training": youth_in_training,
        # Financial inclusion
        "has_mobile_money": has_mobile_money,
        "has_bank_account": has_bank_account,
        "accessed_credit_12m": accessed_credit_12m,
        "credit_source": credit_source,
        "total_savings_usd": total_savings_usd,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "monthly_income_usd", "enterprise_monthly_revenue_usd",
            "vsla_savings_usd", "food_consumption_score",
            "reduced_coping_strategies_index", "business_skills_score",
            "total_savings_usd",
        ],
        rates=[0.07, 0.10, 0.05, 0.04, 0.04, 0.06, 0.08],
        rng=rng,
        mechanism="MAR",
    )
    return df


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _logistic(base, z, slope=1.0):
    """Shift a base probability through a logistic link using scipy expit."""
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6)))
                 + slope * np.asarray(z))
