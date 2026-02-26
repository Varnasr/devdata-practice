"""
Generator 8: Microfinance / Credit Records
───────────────────────────────────────────
Simulates loan-level records from a microfinance institution with borrower
demographics, loan terms, repayment history, and group lending structure.

Rows: one per loan (~25-35k loans).

Realistic features:
  • Group lending (joint liability) structure
  • Repayment schedule with realistic default patterns
  • Interest rates varying by product and risk
  • Loan purpose distribution (agriculture, trade, education, housing)
  • Repeat borrowers with graduation to larger loans
  • Default correlated with loan size, purpose, and borrower characteristics
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, random_dates, COUNTRIES


def generate(n_loans: int = 30000, seed: int = 404) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_loans

    # Loan IDs
    loan_ids = [f"LN-{i:06d}" for i in range(1, n + 1)]
    borrower_ids = [f"BRW-{i:06d}" for i in range(1, n + 1)]

    # Some borrowers are repeat (same borrower ID)
    is_repeat = rng.random(n) < 0.30
    cycle_number = np.ones(n, dtype=int)
    cycle_number[is_repeat] = rng.choice([2, 3, 4, 5], is_repeat.sum(), p=[0.45, 0.30, 0.15, 0.10])
    # Reuse some borrower IDs for repeat borrowers
    repeat_pool = rng.choice(range(n), size=int(is_repeat.sum() * 0.5), replace=True)
    repeat_idx = np.where(is_repeat)[0]
    for i, idx in enumerate(repeat_idx[:len(repeat_pool)]):
        borrower_ids[idx] = borrower_ids[repeat_pool[i]]

    districts, urban = pick_districts(rng, n, urban_share=0.30)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # Borrower characteristics
    female = rng.binomial(1, 0.68, n)  # MFI skew toward women
    age = rng.integers(18, 62, n)
    educ_years = np.clip(rng.normal(6, 3, n), 0, 16).astype(int)
    hh_size = rng.choice(range(1, 10), n)

    # Group lending
    group_id = [f"GRP-{rng.integers(1, 3000):04d}" for _ in range(n)]
    group_size = rng.choice([3, 4, 5, 6, 7], n, p=[0.05, 0.15, 0.40, 0.25, 0.15])
    individual_lending = rng.binomial(1, 0.25 + 0.15 * cycle_number / 5, n)

    # Loan product
    product = rng.choice(
        ["group_micro", "individual_micro", "agri_loan", "sme_loan", "education_loan"],
        n, p=[0.35, 0.25, 0.20, 0.12, 0.08]
    )

    # Loan purpose
    purpose = rng.choice(
        ["trade_retail", "agriculture", "livestock", "housing", "education",
         "health", "consumption", "business_expansion"],
        n, p=[0.25, 0.20, 0.10, 0.08, 0.08, 0.05, 0.09, 0.15]
    )

    # Loan amount (USD, grows with cycle)
    base_amount = np.where(
        product == "sme_loan", rng.lognormal(6.5, 0.5, n),
        np.where(product == "agri_loan", rng.lognormal(5.2, 0.6, n),
                 rng.lognormal(4.8, 0.6, n))
    )
    loan_amount = np.round(base_amount * (1 + 0.3 * (cycle_number - 1)), 2)

    # Interest rate (annual, %)
    base_rate = np.where(product == "group_micro", 24,
                np.where(product == "individual_micro", 22,
                np.where(product == "agri_loan", 18,
                np.where(product == "sme_loan", 16, 20))))
    interest_rate = base_rate + rng.normal(0, 2, n)
    interest_rate = np.round(np.clip(interest_rate, 10, 36), 1)

    # Loan term (months)
    term_months = rng.choice([6, 9, 12, 18, 24], n, p=[0.15, 0.15, 0.35, 0.20, 0.15])

    # Disbursement date
    disbursement_date = random_dates(rng, n, "2019-01-01", "2024-06-30")

    # Collateral
    has_collateral = rng.binomial(1, np.where(individual_lending, 0.55, 0.10))
    collateral_type = np.where(
        has_collateral,
        rng.choice(["savings_deposit", "household_asset", "land_title", "group_guarantee"],
                   n),
        "none"
    )

    # Repayment behavior
    # Default probability (correlated with amount, purpose, cycle)
    default_logit = (
        -2.5
        + 0.3 * np.log(loan_amount) / 10
        - 0.15 * female
        - 0.10 * educ_years / 10
        - 0.25 * (cycle_number > 1).astype(float)
        + 0.20 * (purpose == "consumption").astype(float)
        - 0.15 * has_collateral
        + rng.normal(0, 0.3, n)
    )
    default_prob = 1 / (1 + np.exp(-default_logit))
    defaulted = rng.binomial(1, default_prob)

    # Days past due (for non-defaults too — some are late)
    days_past_due = np.zeros(n, dtype=int)
    late_mask = rng.random(n) < 0.25  # 25% have some late payment
    days_past_due[late_mask] = np.clip(rng.exponential(15, late_mask.sum()), 1, 90).astype(int)
    days_past_due[defaulted.astype(bool)] = np.clip(
        rng.exponential(45, defaulted.sum()), 30, 365
    ).astype(int)

    # Repayment rate (% of expected payments made)
    repayment_rate = np.where(
        defaulted, np.clip(rng.beta(2, 5, n), 0, 0.8),
        np.clip(rng.beta(8, 1, n), 0.5, 1.0)
    )

    # Amount repaid
    expected_repayment = loan_amount * (1 + interest_rate / 100 * term_months / 12)
    amount_repaid = np.round(expected_repayment * repayment_rate, 2)

    # Write-off
    written_off = (defaulted & (days_past_due > 180)).astype(int)

    # Savings balance (for group members)
    savings_balance = np.round(
        np.where(individual_lending, 0, rng.exponential(loan_amount * 0.08)), 2
    )

    # Credit score (internal, 300-850)
    credit_score = np.clip(
        550 + 30 * cycle_number - 80 * defaulted + 20 * female
        + 10 * educ_years + rng.normal(0, 40, n),
        300, 850
    ).astype(int)

    df = pd.DataFrame({
        "loan_id": loan_ids,
        "borrower_id": borrower_ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "education_years": educ_years,
        "household_size": hh_size,
        "group_id": group_id,
        "group_size": group_size,
        "individual_lending": individual_lending,
        "loan_product": product,
        "loan_purpose": purpose,
        "loan_amount_usd": np.round(loan_amount, 2),
        "interest_rate_annual_pct": interest_rate,
        "term_months": term_months,
        "disbursement_date": disbursement_date,
        "cycle_number": cycle_number,
        "has_collateral": has_collateral,
        "collateral_type": collateral_type,
        "defaulted": defaulted,
        "days_past_due": days_past_due,
        "repayment_rate": np.round(repayment_rate, 3),
        "amount_repaid_usd": amount_repaid,
        "written_off": written_off,
        "savings_balance_usd": savings_balance,
        "internal_credit_score": credit_score,
    })

    df = inject_missing(df,
        columns=["education_years", "savings_balance_usd", "household_size"],
        rates=[0.03, 0.05, 0.02],
        rng=rng, mechanism="MCAR")
    return df
