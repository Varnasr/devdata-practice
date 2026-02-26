"""
Generator: Animal Welfare
─────────────────────────
Simulates a community-level animal welfare assessment dataset covering
livestock husbandry practices, veterinary access, working animal welfare,
companion animal populations, and programme interventions.

Rows: ~15k animals/households.

Realistic features:
  • Five Freedoms scoring framework
  • Body condition scoring (BCS 1-5)
  • Working animal welfare (donkeys, horses, oxen)
  • Veterinary access and treatment
  • Rabies vaccination coverage
  • Animal-source food linkages to nutrition
  • Programme intervention effects (training, vet camps)
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, household_ids


def generate(n_records: int = 15000, seed: int = 705) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_records

    ids = [f"ANI-{i:06d}" for i in range(1, n + 1)]
    hh_ids = household_ids(rng, n, prefix="AHH")
    districts, urban = pick_districts(rng, n, urban_share=0.18)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # Animal type
    animal_type = rng.choice(
        ["cattle", "goats", "sheep", "poultry", "donkey", "horse", "pig",
         "camel", "dog", "cat"],
        n, p=[0.20, 0.15, 0.10, 0.18, 0.08, 0.04, 0.08, 0.03, 0.08, 0.06]
    )
    is_livestock = np.isin(animal_type, ["cattle", "goats", "sheep", "poultry", "pig", "camel"])
    is_working = np.isin(animal_type, ["donkey", "horse", "camel"])
    is_companion = np.isin(animal_type, ["dog", "cat"])

    # Owner characteristics
    owner_female = rng.binomial(1, np.where(np.isin(animal_type, ["poultry", "goats"]), 0.55, 0.25))
    owner_educ = np.clip(rng.normal(5, 3.5, n), 0, 16).astype(int)
    wealth = rng.normal(0, 1, n)

    # Programme participation
    received_training = rng.binomial(1, 0.25, n)
    attended_vet_camp = rng.binomial(1, 0.18, n)
    in_community_animal_health = rng.binomial(1, 0.15, n)

    # --- Five Freedoms Assessment (each 1-5 scale) ---
    base_welfare = 2.5 + 0.3 * wealth + 0.4 * received_training + 0.2 * owner_educ / 18

    freedom_hunger = np.clip(rng.normal(base_welfare + 0.3, 0.8, n), 1, 5).round(1)
    freedom_discomfort = np.clip(rng.normal(base_welfare, 0.9, n), 1, 5).round(1)
    freedom_pain = np.clip(rng.normal(base_welfare - 0.2, 0.9, n), 1, 5).round(1)
    freedom_behavior = np.clip(rng.normal(base_welfare + 0.1, 0.8, n), 1, 5).round(1)
    freedom_fear = np.clip(rng.normal(base_welfare + 0.2, 0.8, n), 1, 5).round(1)
    welfare_score = np.round((freedom_hunger + freedom_discomfort + freedom_pain +
                              freedom_behavior + freedom_fear) / 5, 2)

    # Body condition score (1-5, 1=emaciated, 5=obese)
    bcs = np.clip(rng.normal(2.8 + 0.2 * wealth + 0.1 * received_training, 0.6, n), 1, 5).round(1)

    # Shelter quality (1-5)
    shelter_score = np.clip(
        rng.normal(2.5 + 0.4 * wealth + 0.3 * received_training, 0.8, n), 1, 5
    ).round(1)

    # Veterinary access
    distance_to_vet_km = np.clip(rng.exponential(np.where(urban, 3, 12)), 0.5, 60).round(1)
    accessed_vet_last_year = rng.binomial(1, _logistic(0.30, wealth + 0.15 * attended_vet_camp - 0.02 * distance_to_vet_km, 0.3))
    vaccinated = rng.binomial(1, _logistic(0.35, wealth + 0.2 * attended_vet_camp + 0.1 * in_community_animal_health, 0.3))
    dewormed = rng.binomial(1, _logistic(0.30, wealth + 0.15 * received_training, 0.3))

    # Disease in past year
    had_disease = rng.binomial(1, _logistic(0.25, -welfare_score / 5 + 0.1, 0.4))
    disease_type = np.where(had_disease,
        rng.choice(["respiratory", "gastrointestinal", "parasitic", "reproductive", "foot_rot",
                    "rabies_suspect", "other"], n,
                   p=[0.20, 0.18, 0.22, 0.10, 0.12, 0.05, 0.13]),
        "none")

    # Mortality (last year per household)
    animal_mortality_rate = np.clip(rng.beta(1.5, 8, n), 0, 0.5).round(3)

    # Working animal specific
    hours_worked_daily = np.where(is_working,
        np.clip(rng.normal(6, 2, n), 1, 14).round(1), np.nan)
    has_wounds = np.where(is_working,
        rng.binomial(1, _logistic(0.30, -welfare_score / 5, 0.4)), np.nan)
    uses_proper_harness = np.where(is_working,
        rng.binomial(1, _logistic(0.35, wealth + 0.2 * received_training, 0.3)), np.nan)

    # Companion animal specific
    rabies_vaccinated = np.where(is_companion,
        rng.binomial(1, _logistic(0.25, wealth + 0.2 * urban.astype(float), 0.4)), np.nan)
    sterilized = np.where(is_companion,
        rng.binomial(1, _logistic(0.12, wealth + 0.15 * urban.astype(float), 0.4)), np.nan)

    # Livestock productivity linkage
    animal_source_food_consumption = np.where(is_livestock,
        rng.binomial(1, _logistic(0.55, wealth + 0.1 * bcs / 5, 0.3)), np.nan)

    df = pd.DataFrame({
        "record_id": ids, "household_id": hh_ids,
        "country": countries, "district": districts, "urban": urban.astype(int),
        "animal_type": animal_type,
        "owner_female": owner_female, "owner_education_years": owner_educ,
        "received_welfare_training": received_training,
        "attended_vet_camp": attended_vet_camp,
        "in_community_animal_health_prog": in_community_animal_health,
        "freedom_hunger_thirst": freedom_hunger,
        "freedom_discomfort": freedom_discomfort,
        "freedom_pain_injury": freedom_pain,
        "freedom_normal_behavior": freedom_behavior,
        "freedom_fear_distress": freedom_fear,
        "welfare_score_avg": welfare_score,
        "body_condition_score": bcs,
        "shelter_score": shelter_score,
        "distance_to_vet_km": distance_to_vet_km,
        "accessed_vet_last_year": accessed_vet_last_year,
        "vaccinated": vaccinated, "dewormed": dewormed,
        "had_disease_last_year": had_disease, "disease_type": disease_type,
        "animal_mortality_rate": animal_mortality_rate,
        "working_hours_daily": hours_worked_daily,
        "working_has_wounds": has_wounds,
        "working_proper_harness": uses_proper_harness,
        "companion_rabies_vaccinated": rabies_vaccinated,
        "companion_sterilized": sterilized,
        "hh_consumes_animal_source_food": animal_source_food_consumption,
    })

    df = inject_missing(df,
        columns=["welfare_score_avg", "body_condition_score", "distance_to_vet_km"],
        rates=[0.05, 0.04, 0.06],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
