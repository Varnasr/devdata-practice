"""
Generator: BCC / KAP Survey
────────────────────────────
Simulates Knowledge-Attitude-Practice (KAP) survey data for behaviour change
communication programmes across health, WASH, and nutrition domains.

Rows: ~20k individuals.

Realistic features:
  • KAP cascade: campaign exposure improves knowledge > attitudes > practice
  • Multiple campaign modalities (radio, community drama, peer education, etc.)
  • Knowledge, attitude, and practice sub-indicators
  • KAP gap measures (knowledge-practice, attitude-practice)
  • Self-efficacy and social norms
  • MAR missingness mechanism
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_individuals: int = 20000, seed: int = 801) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    # Demographics
    ids = [f"IND-{i:07d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.36)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.52, n)
    age = rng.integers(15, 65, n)
    educ_years = np.clip(
        rng.normal(7 + 2.5 * urban.astype(float), 3.5, n), 0, 18
    ).astype(int)
    wealth_latent = rng.normal(0, 1, n) + 0.3 * urban.astype(float) + 0.05 * educ_years
    wealth_quintile = np.clip(
        (pd.qcut(wealth_latent, 5, labels=False, duplicates="drop") + 1), 1, 5
    ).astype(int)

    # --- Campaign exposure ---
    exposed_to_campaign = rng.binomial(
        1, _logistic(0.40, 0.15 * urban.astype(float) + 0.05 * educ_years / 10, 0.3)
    )
    campaign_types = ["radio", "community_drama", "peer_education", "SMS", "poster", "social_media"]
    campaign_type = np.where(
        exposed_to_campaign,
        rng.choice(campaign_types, n, p=[0.28, 0.18, 0.15, 0.14, 0.13, 0.12]),
        "none",
    )
    campaign_doses = np.where(
        exposed_to_campaign,
        np.clip(rng.poisson(np.clip(2.0 + 0.5 * urban.astype(float), 0.1, 10), n), 1, 5),
        0,
    )

    # Exposure intensity (latent driver for KAP cascade)
    exposure_effect = exposed_to_campaign * campaign_doses / 5.0

    # --- Knowledge indicators ---
    k_base = 0.35 + 0.10 * educ_years / 18 + 0.25 * exposure_effect
    knows_handwashing_times = rng.binomial(1, _logistic(0.50, exposure_effect * 0.8 + educ_years / 20, 0.4))
    knows_ors_for_diarrhea = rng.binomial(1, _logistic(0.40, exposure_effect * 0.7 + educ_years / 25, 0.4))
    knows_exclusive_breastfeeding = rng.binomial(1, _logistic(0.35, exposure_effect * 0.6 + female * 0.15, 0.4))
    knows_family_planning_methods = rng.binomial(1, _logistic(0.45, exposure_effect * 0.7 + educ_years / 20, 0.4))
    knows_hiv_transmission = rng.binomial(1, _logistic(0.50, exposure_effect * 0.6 + educ_years / 18, 0.4))

    knowledge_score = np.clip(
        rng.normal(
            4.0 + 2.5 * exposure_effect + 0.15 * educ_years + 0.2 * wealth_latent,
            1.5, n
        ), 0, 10
    ).round(1)

    # --- Attitude indicators (exposure effect smaller than knowledge) ---
    approves_family_planning = rng.binomial(
        1, _logistic(0.55, exposure_effect * 0.5 + educ_years / 20 + 0.1 * urban.astype(float), 0.3)
    )
    approves_girls_education = rng.binomial(
        1, _logistic(0.65, exposure_effect * 0.4 + educ_years / 18, 0.3)
    )
    gender_equitable_attitude = rng.binomial(
        1, _logistic(0.40, exposure_effect * 0.5 + educ_years / 15 + 0.1 * female, 0.3)
    )
    stigma_hiv = rng.binomial(
        1, _logistic(0.35, -exposure_effect * 0.4 - educ_years / 20, 0.3)
    )
    stigma_mental_health = rng.binomial(
        1, _logistic(0.45, -exposure_effect * 0.3 - educ_years / 25, 0.3)
    )

    attitude_score = np.clip(
        rng.normal(
            4.5 + 1.8 * exposure_effect + 0.12 * educ_years + 0.15 * wealth_latent,
            1.4, n
        ), 0, 10
    ).round(1)

    # --- Practice indicators (exposure effect smallest — realistic KAP cascade) ---
    practices_handwashing = rng.binomial(
        1, _logistic(0.35, exposure_effect * 0.3 + 0.1 * wealth_latent, 0.3)
    )
    uses_treated_water = rng.binomial(
        1, _logistic(0.25, exposure_effect * 0.25 + 0.15 * urban.astype(float) + 0.05 * wealth_latent, 0.3)
    )
    exclusive_breastfeeding_6m = rng.binomial(
        1, _logistic(0.30, exposure_effect * 0.3 + female * 0.1, 0.3)
    )
    uses_modern_contraception = rng.binomial(
        1, _logistic(0.28, exposure_effect * 0.25 + educ_years / 25 + 0.05 * urban.astype(float), 0.3)
    )
    seeks_antenatal_care = rng.binomial(
        1, _logistic(0.40, exposure_effect * 0.3 + 0.1 * wealth_latent + 0.1 * urban.astype(float), 0.3)
    )
    uses_mosquito_net = rng.binomial(
        1, _logistic(0.35, exposure_effect * 0.25 + 0.08 * wealth_latent, 0.3)
    )

    practice_score = np.clip(
        rng.normal(
            3.5 + 1.2 * exposure_effect + 0.10 * educ_years + 0.1 * wealth_latent,
            1.6, n
        ), 0, 10
    ).round(1)

    # --- KAP gaps ---
    knowledge_practice_gap = np.round(knowledge_score - practice_score, 1)
    attitude_practice_gap = np.round(attitude_score - practice_score, 1)

    # --- Self-efficacy ---
    self_efficacy_score = np.clip(
        rng.normal(
            2.8 + 0.4 * exposure_effect + 0.05 * educ_years + 0.1 * wealth_latent,
            0.7, n
        ), 1, 5
    ).round(1)

    # --- Social norms ---
    perceives_community_support = rng.binomial(
        1, _logistic(0.35, exposure_effect * 0.5 + 0.05 * educ_years / 10, 0.3)
    )
    discussed_with_peers = rng.binomial(
        1, _logistic(0.30, exposure_effect * 0.6 + 0.1 * (age < 35).astype(float), 0.3)
    )

    df = pd.DataFrame({
        "individual_id": ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "education_years": educ_years,
        "wealth_quintile": wealth_quintile,
        "exposed_to_campaign": exposed_to_campaign,
        "campaign_type": campaign_type,
        "campaign_doses": campaign_doses,
        "knowledge_score": knowledge_score,
        "knows_handwashing_times": knows_handwashing_times,
        "knows_ors_for_diarrhea": knows_ors_for_diarrhea,
        "knows_exclusive_breastfeeding": knows_exclusive_breastfeeding,
        "knows_family_planning_methods": knows_family_planning_methods,
        "knows_hiv_transmission": knows_hiv_transmission,
        "attitude_score": attitude_score,
        "approves_family_planning": approves_family_planning,
        "approves_girls_education": approves_girls_education,
        "gender_equitable_attitude": gender_equitable_attitude,
        "stigma_hiv": stigma_hiv,
        "stigma_mental_health": stigma_mental_health,
        "practice_score": practice_score,
        "practices_handwashing": practices_handwashing,
        "uses_treated_water": uses_treated_water,
        "exclusive_breastfeeding_6m": exclusive_breastfeeding_6m,
        "uses_modern_contraception": uses_modern_contraception,
        "seeks_antenatal_care": seeks_antenatal_care,
        "uses_mosquito_net": uses_mosquito_net,
        "knowledge_practice_gap": knowledge_practice_gap,
        "attitude_practice_gap": attitude_practice_gap,
        "self_efficacy_score": self_efficacy_score,
        "perceives_community_support": perceives_community_support,
        "discussed_with_peers": discussed_with_peers,
    })

    df = inject_missing(df,
        columns=["knowledge_score", "attitude_score", "practice_score",
                 "self_efficacy_score", "campaign_doses"],
        rates=[0.04, 0.05, 0.06, 0.07, 0.03],
        rng=rng, mechanism="MAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + slope * np.asarray(z))
