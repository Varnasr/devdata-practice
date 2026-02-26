"""
Generator: Intersectional Inequality
─────────────────────────────────────
Simulates individual-level data for intersectional analysis of caste, gender,
disability, and other identity dimensions with socioeconomic outcomes.

Rows: ~25k individuals.

Realistic features:
  • Caste, gender, religion, disability, sexuality, and indigeneity dimensions
  • Socioeconomic outcomes: income, employment, housing, food security
  • Discrimination experience and context
  • Access to services: education, healthcare, finance, government, justice
  • Social capital and institutional trust
  • Multiplicative (intersectional) disadvantage in outcomes
  • Inclusion programme participation
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, individual_ids


def generate(n_individuals: int = 25000, seed: int = 805) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    ids = individual_ids(rng, n, prefix="INT")
    districts, urban = pick_districts(rng, n, urban_share=0.33)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.52, n)
    age = rng.integers(15, 70, n)
    educ_years = np.clip(rng.normal(6 + 2 * urban.astype(float) - 0.5 * female, 3.5, n), 0, 18).astype(int)

    # --- Identity dimensions ---
    caste_category = rng.choice(
        ["general", "obc", "sc", "st"],
        n, p=[0.28, 0.35, 0.22, 0.15]
    )
    is_sc = (caste_category == "sc").astype(float)
    is_st = (caste_category == "st").astype(float)
    is_marginalized_caste = (is_sc + is_st).clip(0, 1)

    religion = rng.choice(
        ["hindu", "muslim", "christian", "buddhist", "sikh", "other"],
        n, p=[0.55, 0.22, 0.10, 0.05, 0.04, 0.04]
    )
    is_minority_religion = (~np.isin(religion, ["hindu"])).astype(float)

    has_disability = rng.binomial(1, 0.08, n)
    disability_type = np.where(
        has_disability,
        rng.choice(["physical", "visual", "hearing", "cognitive", "multiple"],
                   n, p=[0.30, 0.22, 0.18, 0.15, 0.15]),
        "none"
    )

    sexual_minority = rng.binomial(1, 0.04, n)
    indigenous = rng.binomial(1, 0.10, n)

    # --- Intersectional penalty score (additive then multiplicative) ---
    # Each identity axis contributes; interactions amplify disadvantage
    penalty = (0.3 * is_marginalized_caste
               + 0.25 * female
               + 0.35 * has_disability
               + 0.15 * is_minority_religion
               + 0.15 * sexual_minority
               + 0.15 * indigenous)
    # Multiplicative interactions: being in multiple groups is worse than sum
    n_disadvantages = (is_marginalized_caste + female + has_disability
                       + sexual_minority + indigenous)
    intersectional_multiplier = 1 + 0.15 * np.clip(n_disadvantages - 1, 0, 5)
    penalty = penalty * intersectional_multiplier

    # Wealth proxy
    wealth = rng.normal(0, 1, n) + 0.3 * urban.astype(float) + 0.2 * educ_years / 10 - 0.4 * penalty

    # --- Socioeconomic outcomes ---
    employed = rng.binomial(1, _logistic(0.55, wealth + 0.1 * educ_years / 10 - 0.3 * penalty, 0.4))
    occupation_type = np.where(
        employed,
        rng.choice(["professional", "skilled_manual", "unskilled_manual", "agriculture",
                    "service", "domestic", "casual_labor"],
                   n, p=[0.10, 0.15, 0.20, 0.20, 0.15, 0.10, 0.10]),
        "unemployed"
    )

    monthly_income_usd = np.where(
        employed,
        np.round(np.exp(rng.normal(3.8 + 0.3 * wealth - 0.25 * penalty, 0.5, n)), 2),
        0
    )

    housing_quality_score = np.clip(
        rng.normal(3 + 0.5 * wealth - 0.3 * penalty, 0.8, n), 1, 5
    ).round(0).astype(int)

    food_security_score = np.clip(
        rng.normal(3.5 + 0.4 * wealth - 0.3 * penalty + 0.1 * urban.astype(float), 0.9, n), 1, 5
    ).round(0).astype(int)

    # --- Discrimination experience ---
    experienced_discrimination = rng.binomial(
        1, _logistic(0.20, penalty, 0.6)
    )
    discrimination_basis = np.where(
        experienced_discrimination,
        rng.choice(["caste", "gender", "disability", "religion", "ethnicity", "sexuality"],
                   n, p=[0.30, 0.25, 0.15, 0.13, 0.10, 0.07]),
        "none"
    )
    discrimination_context = np.where(
        experienced_discrimination,
        rng.choice(["employment", "education", "health_facility", "public_space", "housing"],
                   n, p=[0.30, 0.22, 0.20, 0.15, 0.13]),
        "none"
    )

    # --- Access to services ---
    accessed_education = rng.binomial(1, _logistic(0.65, wealth + 0.1 * urban.astype(float) - 0.2 * penalty, 0.4))
    accessed_healthcare = rng.binomial(1, _logistic(0.55, wealth + 0.15 * urban.astype(float) - 0.2 * penalty, 0.4))
    accessed_financial_services = rng.binomial(1, _logistic(0.30, wealth + 0.1 * educ_years / 10 - 0.25 * penalty, 0.5))
    accessed_government_schemes = rng.binomial(1, _logistic(0.35, 0.1 * educ_years / 10 + 0.1 * urban.astype(float) - 0.15 * penalty, 0.4))
    accessed_justice = rng.binomial(1, _logistic(0.20, wealth + 0.1 * educ_years / 10 - 0.2 * penalty, 0.4))

    # --- Social capital ---
    community_group_member = rng.binomial(1, _logistic(0.30, 0.1 * educ_years / 10 - 0.1 * penalty, 0.3))
    trust_in_institutions = np.clip(
        rng.normal(3.0 - 0.3 * penalty + 0.1 * wealth, 0.9, n), 1, 5
    ).round(0).astype(int)
    political_participation = rng.binomial(
        1, _logistic(0.40, 0.1 * educ_years / 10 + 0.05 * (age > 18).astype(float) - 0.1 * penalty, 0.3)
    )

    # --- Programme ---
    received_inclusion_programme = rng.binomial(1, 0.22, n)
    programme_type = np.where(
        received_inclusion_programme,
        rng.choice(["affirmative_action", "skills_training", "cash_transfer",
                    "awareness_campaign", "legal_empowerment", "self_help_group"],
                   n, p=[0.18, 0.20, 0.18, 0.16, 0.13, 0.15]),
        "none"
    )

    df = pd.DataFrame({
        "individual_id": ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "female": female, "age": age,
        "education_years": educ_years,
        "caste_category": caste_category, "religion": religion,
        "has_disability": has_disability, "disability_type": disability_type,
        "sexual_minority": sexual_minority, "indigenous": indigenous,
        "monthly_income_usd": monthly_income_usd,
        "employed": employed, "occupation_type": occupation_type,
        "housing_quality_score": housing_quality_score,
        "food_security_score": food_security_score,
        "experienced_discrimination": experienced_discrimination,
        "discrimination_basis": discrimination_basis,
        "discrimination_context": discrimination_context,
        "accessed_education": accessed_education,
        "accessed_healthcare": accessed_healthcare,
        "accessed_financial_services": accessed_financial_services,
        "accessed_government_schemes": accessed_government_schemes,
        "accessed_justice": accessed_justice,
        "community_group_member": community_group_member,
        "trust_in_institutions": trust_in_institutions,
        "political_participation": political_participation,
        "received_inclusion_programme": received_inclusion_programme,
        "programme_type": programme_type,
    })

    df = inject_missing(df,
        columns=["monthly_income_usd", "education_years", "trust_in_institutions", "housing_quality_score"],
        rates=[0.06, 0.04, 0.05, 0.03],
        rng=rng, mechanism="MAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
