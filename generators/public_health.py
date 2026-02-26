"""
Generator: Public Health & Epidemiology
────────────────────────────────────────
Simulates individual-level public health surveillance data covering disease
prevalence, health facility utilisation, insurance enrolment, NCD screening,
mental health, and COVID-19 vaccination — the kind produced by population-based
health surveys (e.g. MIS, PHIA, STEPS) and HMIS extracts.

Rows: one per individual (~20k).

Realistic features:
  • Disease surveillance (malaria, TB, HIV) with age/sex/wealth gradients
  • Health facility visits and out-of-pocket (OOP) spending with catastrophic
    expenditure flag (>10 % of household consumption)
  • Community health worker (CHW) contact and referral cascade
  • Health insurance enrolment (CBHI, NHIF) driven by wealth and urban status
  • NCD risk factors — hypertension screening, diabetes screening
  • PHQ-9-like depression screening score (0-27)
  • COVID-19 vaccination (dose 0 / 1 / 2 / booster)
  • Wealth quintile as a cross-cutting equity axis
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, household_ids


def generate(n_individuals: int = 20000, seed: int = 501) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    # --- IDs & geography ---
    ind_ids = [f"IND-{i:07d}" for i in range(1, n + 1)]
    hh_ids = household_ids(rng, n, prefix="HH")  # one row per individual; HH id reused below
    districts, urban = pick_districts(rng, n, urban_share=0.30)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # Cluster some individuals into shared households (~4 per HH on average)
    n_hh = n // 4
    hh_pool = household_ids(rng, n_hh, prefix="HH")
    hh_assignment = rng.choice(hh_pool, n)

    # --- Demographics ---
    female = rng.binomial(1, 0.52, n)
    age = np.clip(rng.exponential(25, n) + 1, 0, 95).astype(int)
    # Re-draw children more realistically
    child_mask = rng.random(n) < 0.18
    age[child_mask] = rng.integers(0, 15, child_mask.sum())

    # --- Wealth quintile (1-5) ---
    wealth_score = rng.normal(0, 1, n) + 0.5 * urban.astype(float)
    wealth_quintile = pd.qcut(wealth_score, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Education years (for adults)
    educ = np.where(
        age >= 6,
        np.clip(rng.normal(5 + 1.5 * (wealth_quintile - 1) + 1.5 * urban.astype(float), 3, n), 0, 18).astype(int),
        0,
    )
    educ = np.minimum(educ, np.maximum(age - 5, 0))

    # Monthly household consumption per capita (USD PPP)
    monthly_pce = np.round(np.exp(3.5 + 0.45 * (wealth_quintile - 3) / 2 + rng.normal(0, 0.35, n)), 2)

    # ------------------------------------------------------------------ #
    # Disease surveillance
    # ------------------------------------------------------------------ #

    # Malaria — recent infection (RDT+), higher in rural, children, low wealth
    malaria_prob = _logistic(0.15, -(wealth_quintile - 3) / 2 - 0.4 * urban.astype(float)
                             + 0.3 * (age < 5).astype(float), slope=0.5)
    malaria_rdt_positive = rng.binomial(1, malaria_prob)
    malaria_tested = rng.binomial(1, _logistic(0.45, (wealth_quintile - 3) / 2 + 0.3 * urban.astype(float), 0.4))
    # If not tested, RDT result unknown
    malaria_rdt_positive = np.where(malaria_tested, malaria_rdt_positive, -1)

    # TB — ever diagnosed, currently on treatment
    tb_ever = rng.binomial(1, _logistic(0.03, -(wealth_quintile - 3) / 3, 0.4))
    tb_on_treatment = np.where(tb_ever, rng.binomial(1, _logistic(0.60, (wealth_quintile - 3) / 3, 0.4)), 0)

    # HIV — status known, positive, on ART
    hiv_tested_ever = rng.binomial(1, _logistic(0.55, (wealth_quintile - 3) / 3 + 0.2 * urban.astype(float)
                                                 + 0.15 * female, 0.4))
    hiv_positive = rng.binomial(1, _logistic(0.06, -(wealth_quintile - 3) / 4
                                              + 0.15 * female
                                              + 0.2 * ((age >= 15) & (age <= 49)).astype(float), 0.3))
    hiv_on_art = np.where(hiv_positive & hiv_tested_ever,
                          rng.binomial(1, _logistic(0.75, (wealth_quintile - 3) / 3, 0.3)), 0)

    # ------------------------------------------------------------------ #
    # Health facility utilisation & spending
    # ------------------------------------------------------------------ #
    facility_visits_12m = np.clip(
        rng.poisson(np.clip(1.8 + 0.5 * (wealth_quintile - 3) / 2 + 0.6 * urban.astype(float), 0.1, 10)),
        0, 24,
    )
    oop_spending_usd = np.where(
        facility_visits_12m > 0,
        np.round(rng.exponential(15 + 5 * (wealth_quintile - 1)) * facility_visits_12m / 3, 2),
        0.0,
    )
    catastrophic_health_exp = (oop_spending_usd > 0.10 * monthly_pce * 12).astype(int)

    # ------------------------------------------------------------------ #
    # CHW contact & referral
    # ------------------------------------------------------------------ #
    chw_contact_6m = rng.binomial(1, _logistic(0.30, -(wealth_quintile - 3) / 3
                                                - 0.2 * urban.astype(float), 0.3))
    chw_referred = np.where(chw_contact_6m,
                            rng.binomial(1, _logistic(0.35, (age < 5).astype(float) * 0.3, 0.3)), 0)
    chw_referral_completed = np.where(chw_referred,
                                      rng.binomial(1, _logistic(0.55, (wealth_quintile - 3) / 3, 0.3)), 0)

    # ------------------------------------------------------------------ #
    # Health insurance
    # ------------------------------------------------------------------ #
    insurance_type_pool = ["none", "CBHI", "NHIF", "private", "employer"]
    ins_probs = np.column_stack([
        _logistic(0.55, -(wealth_quintile - 3) / 2 - 0.15 * urban.astype(float), 0.4),  # none
        _logistic(0.18, -(wealth_quintile - 3) / 4, 0.3),   # CBHI (rural, low-mid)
        _logistic(0.12, (wealth_quintile - 3) / 3 + 0.2 * urban.astype(float), 0.4),  # NHIF
        _logistic(0.06, (wealth_quintile - 3) / 2, 0.5),  # private
        _logistic(0.04, (wealth_quintile - 3) / 2 + 0.3 * urban.astype(float), 0.5),  # employer
    ])
    ins_probs = ins_probs / ins_probs.sum(axis=1, keepdims=True)
    insurance_enrolled = np.array([rng.choice(insurance_type_pool, p=ins_probs[i]) for i in range(n)])

    # ------------------------------------------------------------------ #
    # NCD risk factors (adults 18+)
    # ------------------------------------------------------------------ #
    adult = (age >= 18).astype(float)

    # Hypertension screening
    bp_screened = np.where(adult,
                           rng.binomial(1, _logistic(0.30, (wealth_quintile - 3) / 3
                                                      + 0.2 * urban.astype(float), 0.4)), 0)
    hypertension_dx = np.where(
        bp_screened,
        rng.binomial(1, _logistic(0.22, 0.02 * (age - 40) + 0.1 * (1 - female), 0.3)),
        -1,  # not screened
    )
    hypertension_controlled = np.where(
        hypertension_dx == 1,
        rng.binomial(1, _logistic(0.25, (wealth_quintile - 3) / 3, 0.3)),
        0,
    )

    # Diabetes screening
    diabetes_screened = np.where(adult,
                                 rng.binomial(1, _logistic(0.18, (wealth_quintile - 3) / 3
                                                            + 0.15 * urban.astype(float), 0.4)), 0)
    diabetes_dx = np.where(
        diabetes_screened,
        rng.binomial(1, _logistic(0.08, 0.015 * (age - 40), 0.3)),
        -1,
    )

    # ------------------------------------------------------------------ #
    # Mental health — PHQ-9-like score (0-27)
    # ------------------------------------------------------------------ #
    # 9 items scored 0-3, correlated with poverty, female, shocks
    phq9_latent = (
        -0.3
        - 0.25 * (wealth_quintile - 3) / 2
        + 0.3 * female
        + 0.01 * np.maximum(age - 50, 0)
        + rng.normal(0, 0.8, n)
    )
    phq9_score = np.clip(np.round(9 * _logistic(0.33, phq9_latent, 0.6) * 3), 0, 27).astype(int)
    depression_moderate = (phq9_score >= 10).astype(int)
    depression_severe = (phq9_score >= 20).astype(int)

    # ------------------------------------------------------------------ #
    # COVID-19 vaccination
    # ------------------------------------------------------------------ #
    covid_doses_prob = np.column_stack([
        _logistic(0.25, -(wealth_quintile - 3) / 2, 0.3),  # 0 doses
        _logistic(0.20, 0, 0.1) * np.ones(n),              # 1 dose
        _logistic(0.30, (wealth_quintile - 3) / 3, 0.3),   # 2 doses
        _logistic(0.15, (wealth_quintile - 3) / 2 + 0.2 * urban.astype(float), 0.4),  # booster
    ])
    covid_doses_prob = covid_doses_prob / covid_doses_prob.sum(axis=1, keepdims=True)
    # Only eligible (age >= 12)
    covid_eligible = (age >= 12).astype(int)
    covid_doses = np.where(
        covid_eligible,
        np.array([rng.choice([0, 1, 2, 3], p=covid_doses_prob[i]) for i in range(n)]),
        0,
    )
    covid_vaccinated = (covid_doses >= 1).astype(int)

    # ------------------------------------------------------------------ #
    # Survey metadata
    # ------------------------------------------------------------------ #
    survey_round = rng.choice([1, 2, 3], n, p=[0.35, 0.35, 0.30])

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "individual_id": ind_ids,
        "household_id": hh_assignment,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "survey_round": survey_round,
        "female": female,
        "age": age,
        "education_years": educ,
        "wealth_quintile": wealth_quintile,
        "monthly_pce_usd": monthly_pce,
        # Disease surveillance
        "malaria_tested": malaria_tested,
        "malaria_rdt_positive": malaria_rdt_positive,
        "tb_ever_diagnosed": tb_ever,
        "tb_on_treatment": tb_on_treatment,
        "hiv_tested_ever": hiv_tested_ever,
        "hiv_positive": hiv_positive,
        "hiv_on_art": hiv_on_art,
        # Facility & spending
        "facility_visits_12m": facility_visits_12m,
        "oop_health_spending_usd": oop_spending_usd,
        "catastrophic_health_expenditure": catastrophic_health_exp,
        # CHW
        "chw_contact_6m": chw_contact_6m,
        "chw_referred": chw_referred,
        "chw_referral_completed": chw_referral_completed,
        # Insurance
        "health_insurance_type": insurance_enrolled,
        # NCD
        "bp_screened": bp_screened,
        "hypertension_diagnosed": hypertension_dx,
        "hypertension_controlled": hypertension_controlled,
        "diabetes_screened": diabetes_screened,
        "diabetes_diagnosed": diabetes_dx,
        # Mental health
        "phq9_score": phq9_score,
        "depression_moderate": depression_moderate,
        "depression_severe": depression_severe,
        # COVID-19
        "covid_vaccine_doses": covid_doses,
        "covid_vaccinated": covid_vaccinated,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "oop_health_spending_usd", "phq9_score", "education_years",
            "monthly_pce_usd", "facility_visits_12m", "bp_screened",
        ],
        rates=[0.08, 0.06, 0.03, 0.05, 0.04, 0.03],
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
