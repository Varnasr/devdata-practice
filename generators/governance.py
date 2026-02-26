"""
Generator: Governance & Accountability
───────────────────────────────────────
Simulates data for governance programmes: citizen perception surveys,
public service delivery scorecards, budget transparency, corruption
experience, and social accountability mechanisms.

Rows: ~15k citizens + service delivery observations.

Realistic features:
  • Citizen satisfaction with public services (health, education, water, roads)
  • Corruption experience (bribery incidence, amount)
  • Community scorecard / social audit data
  • Budget literacy and participation
  • Trust in institutions
  • Grievance redress mechanisms
  • Information access (RTI, open data)
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_citizens: int = 15000, seed: int = 711) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_citizens

    ids = [f"CIT-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.38)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.50, n)
    age = rng.integers(18, 70, n)
    educ_years = np.clip(rng.normal(7 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    wealth = rng.normal(0, 1, n) + 0.3 * urban.astype(float)

    # Civic engagement
    voted_last_election = rng.binomial(1, _logistic(0.58, 0.05 * educ_years / 10, 0.3))
    attended_public_meeting = rng.binomial(1, _logistic(0.25, 0.1 * educ_years / 10, 0.3))
    contacted_local_official = rng.binomial(1, _logistic(0.15, wealth + 0.1 * educ_years / 10, 0.3))
    participated_in_budget_process = rng.binomial(1, _logistic(0.08, 0.1 * educ_years / 10, 0.3))

    # Programme
    in_social_accountability_prog = rng.binomial(1, 0.20, n)
    attended_scorecard_session = rng.binomial(1, 0.15 + 0.10 * in_social_accountability_prog)
    used_grievance_mechanism = rng.binomial(1, _logistic(0.10, wealth + 0.15 * in_social_accountability_prog, 0.3))

    # --- Service delivery satisfaction (1-5 scale) ---
    base_sat = 2.8 + 0.2 * wealth + 0.15 * urban.astype(float)
    satisfaction_health = np.clip(rng.normal(base_sat, 0.8, n), 1, 5).round(1)
    satisfaction_education = np.clip(rng.normal(base_sat + 0.1, 0.8, n), 1, 5).round(1)
    satisfaction_water = np.clip(rng.normal(base_sat - 0.2, 0.9, n), 1, 5).round(1)
    satisfaction_roads = np.clip(rng.normal(base_sat - 0.3, 0.9, n), 1, 5).round(1)
    satisfaction_police = np.clip(rng.normal(base_sat - 0.4, 1.0, n), 1, 5).round(1)
    overall_satisfaction = np.round(
        (satisfaction_health + satisfaction_education + satisfaction_water +
         satisfaction_roads + satisfaction_police) / 5, 2
    )

    # --- Trust in institutions (1-5) ---
    trust_local_govt = np.clip(rng.normal(2.8 + 0.1 * in_social_accountability_prog, 0.9, n), 1, 5).round(1)
    trust_national_govt = np.clip(rng.normal(2.6, 1.0, n), 1, 5).round(1)
    trust_judiciary = np.clip(rng.normal(2.5, 1.0, n), 1, 5).round(1)
    trust_police = np.clip(rng.normal(2.3, 1.1, n), 1, 5).round(1)
    trust_traditional_leaders = np.clip(rng.normal(3.2 - 0.2 * urban.astype(float), 0.9, n), 1, 5).round(1)
    trust_ngos = np.clip(rng.normal(3.3, 0.8, n), 1, 5).round(1)

    # --- Corruption experience ---
    bribery_experience = rng.binomial(1, _logistic(0.28, -wealth * 0.1, 0.2))
    bribery_context = np.where(bribery_experience,
        rng.choice(["health_facility", "police", "school", "land_registry", "court",
                    "permit_license", "utility_connection"],
                   n, p=[0.18, 0.22, 0.10, 0.15, 0.12, 0.13, 0.10]),
        "none")
    bribe_amount_usd = np.where(bribery_experience,
        np.round(rng.lognormal(1.5 + 0.3 * wealth, 0.8, n), 2), 0)
    reported_bribery = np.where(bribery_experience,
        rng.binomial(1, 0.08 + 0.05 * in_social_accountability_prog), 0)
    perceives_corruption_worsening = rng.binomial(1, 0.45 - 0.05 * in_social_accountability_prog)

    # --- Budget transparency ---
    aware_of_local_budget = rng.binomial(1, _logistic(0.25, 0.15 * educ_years / 10 + 0.2 * in_social_accountability_prog, 0.4))
    accessed_budget_info = np.where(aware_of_local_budget,
        rng.binomial(1, 0.35 + 0.15 * in_social_accountability_prog), 0)
    budget_literacy_score = np.clip(
        rng.normal(3 + 0.8 * educ_years / 10 + 0.5 * accessed_budget_info, 1.5, n), 0, 10
    ).round(1)

    # --- Information access ---
    knows_rti = rng.binomial(1, _logistic(0.15, 0.15 * educ_years / 10 + 0.1 * urban.astype(float), 0.4))
    used_rti = np.where(knows_rti, rng.binomial(1, 0.12), 0)
    gets_info_radio = rng.binomial(1, 0.55, n)
    gets_info_social_media = rng.binomial(1, _logistic(0.30, 0.2 * urban.astype(float) + 0.1 * (age < 35).astype(float), 0.4))
    gets_info_community_meeting = rng.binomial(1, 0.30 - 0.10 * urban.astype(float))

    # Service delivery scorecard (community-level aggregated to individual)
    scorecard_health = np.clip(rng.normal(55, 15, n), 0, 100).round(0).astype(int)
    scorecard_education = np.clip(rng.normal(58, 14, n), 0, 100).round(0).astype(int)
    scorecard_water = np.clip(rng.normal(48, 18, n), 0, 100).round(0).astype(int)

    df = pd.DataFrame({
        "citizen_id": ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "female": female, "age": age,
        "education_years": educ_years,
        "voted_last_election": voted_last_election,
        "attended_public_meeting": attended_public_meeting,
        "contacted_local_official": contacted_local_official,
        "participated_in_budget_process": participated_in_budget_process,
        "in_social_accountability_prog": in_social_accountability_prog,
        "attended_scorecard_session": attended_scorecard_session,
        "used_grievance_mechanism": used_grievance_mechanism,
        "satisfaction_health": satisfaction_health,
        "satisfaction_education": satisfaction_education,
        "satisfaction_water": satisfaction_water,
        "satisfaction_roads": satisfaction_roads,
        "satisfaction_police": satisfaction_police,
        "overall_service_satisfaction": overall_satisfaction,
        "trust_local_govt": trust_local_govt,
        "trust_national_govt": trust_national_govt,
        "trust_judiciary": trust_judiciary,
        "trust_police": trust_police,
        "trust_traditional_leaders": trust_traditional_leaders,
        "trust_ngos": trust_ngos,
        "bribery_experience": bribery_experience,
        "bribery_context": bribery_context,
        "bribe_amount_usd": bribe_amount_usd,
        "reported_bribery": reported_bribery,
        "perceives_corruption_worsening": perceives_corruption_worsening,
        "aware_of_local_budget": aware_of_local_budget,
        "budget_literacy_score": budget_literacy_score,
        "knows_rti_law": knows_rti,
        "gets_info_radio": gets_info_radio,
        "gets_info_social_media": gets_info_social_media,
        "scorecard_health": scorecard_health,
        "scorecard_education": scorecard_education,
        "scorecard_water": scorecard_water,
    })

    df = inject_missing(df,
        columns=["bribe_amount_usd", "budget_literacy_score", "overall_service_satisfaction"],
        rates=[0.08, 0.05, 0.03],
        rng=rng, mechanism="MNAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
