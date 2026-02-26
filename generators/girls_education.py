"""
Generator: Girls' Education
────────────────────────────
Simulates a longitudinal girls' education programme dataset tracking
enrollment, attendance, learning outcomes, transition rates, menstrual
hygiene management, safety, and barriers to schooling.

Rows: ~20k girls (individual-level, grades 1-12).

Realistic features:
  • Enrollment & dropout with gendered barriers (marriage, pregnancy, distance)
  • Menstrual hygiene management (MHM) affecting attendance
  • Safety perceptions (GBV on route to school, SRGBV)
  • Scholarship/stipend programme effects
  • Transition rates (primary → secondary)
  • Parental attitudes toward girls' education
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_girls: int = 20000, seed: int = 702) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_girls

    ids = [f"GIRL-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.28)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    age = rng.integers(6, 19, n)
    grade = np.clip(age - 5 - rng.choice([0, 0, 0, 1, 1, 2], n, p=[0.4, 0.2, 0.15, 0.12, 0.08, 0.05]), 1, 12)

    # SES
    wealth_quintile = rng.choice([1, 2, 3, 4, 5], n, p=[0.25, 0.22, 0.20, 0.18, 0.15])
    ses = (wealth_quintile - 3) / 2 + rng.normal(0, 0.3, n)

    # Household
    mother_educ = np.clip(rng.normal(3 + 1.5 * (wealth_quintile - 1), 3, n), 0, 16).astype(int)
    father_alive = rng.binomial(1, 0.90, n)
    mother_alive = rng.binomial(1, 0.93, n)
    n_siblings = rng.poisson(3, n)
    hh_size = np.clip(n_siblings + 2, 2, 14)
    parent_values_girls_edu = rng.binomial(1, _logistic(0.55, ses + 0.1 * mother_educ / 10, 0.5))

    # Programme
    receives_scholarship = rng.binomial(1, 0.25 + 0.10 * (wealth_quintile <= 2).astype(float))
    receives_school_meals = rng.binomial(1, 0.35, n)
    in_safe_spaces_programme = rng.binomial(1, 0.20, n)

    # Distance to school
    distance_km = np.clip(rng.exponential(2 if True else 5, n) + (0 if True else 3), 0.1, 15)
    for i in range(n):
        if urban[i]:
            distance_km[i] = np.clip(rng.exponential(1.5), 0.1, 8)
        else:
            distance_km[i] = np.clip(rng.exponential(4), 0.3, 15)

    # Safety
    feels_safe_route = rng.binomial(1, _logistic(0.60, ses + 0.3 * urban.astype(float) - 0.02 * distance_km, 0.3))
    experienced_harassment = rng.binomial(1, _logistic(0.15, -ses + 0.01 * distance_km, 0.3))
    srgbv_experienced = rng.binomial(1, _logistic(0.10, -ses, 0.2))

    # MHM (for age >= 10)
    has_menstruated = (age >= rng.choice([10, 11, 11, 12, 12, 13, 13, 14], n)).astype(int)
    mhm_knowledge = np.where(has_menstruated,
                              rng.binomial(1, _logistic(0.55, ses + 0.2 * in_safe_spaces_programme, 0.4)), 0)
    has_sanitary_products = np.where(has_menstruated,
                                     rng.binomial(1, _logistic(0.40, ses + 0.15 * receives_scholarship, 0.4)), 0)
    missed_school_menstruation = np.where(
        has_menstruated & ~has_sanitary_products.astype(bool),
        rng.binomial(1, 0.55), 0
    )

    # Enrollment & attendance
    enrolled = rng.binomial(1, _logistic(
        0.85,
        ses + 0.15 * receives_scholarship + 0.10 * parent_values_girls_edu
        - 0.03 * distance_km - 0.15 * (age >= 14).astype(float)
        + 0.08 * urban.astype(float),
        0.5
    ))
    attendance_rate = np.where(enrolled,
        np.clip(
            rng.beta(6, 1.5, n)
            - 0.05 * missed_school_menstruation
            - 0.02 * distance_km / 10
            + 0.03 * receives_school_meals
            + 0.02 * feels_safe_route,
            0.3, 1.0
        ), 0)

    # Learning (test scores 0-100)
    math_score = np.where(enrolled,
        np.clip(
            40 + 8 * ses + 3 * mother_educ / 10 + 15 * attendance_rate
            + 3 * receives_scholarship + rng.normal(0, 10, n), 0, 100
        ).astype(int), np.nan)
    literacy_score = np.where(enrolled,
        np.clip(
            45 + 7 * ses + 4 * mother_educ / 10 + 14 * attendance_rate
            + 4 * in_safe_spaces_programme + rng.normal(0, 10, n), 0, 100
        ).astype(int), np.nan)

    # Dropout & barriers
    dropped_out = (~enrolled.astype(bool)).astype(int)
    barrier_marriage = np.where(dropped_out & (age >= 13), rng.binomial(1, 0.25), 0)
    barrier_pregnancy = np.where(dropped_out & (age >= 14), rng.binomial(1, 0.12), 0)
    barrier_cost = np.where(dropped_out, rng.binomial(1, 0.35 - 0.10 * receives_scholarship), 0)
    barrier_distance = np.where(dropped_out, rng.binomial(1, np.clip(0.05 * distance_km, 0, 0.5)), 0)
    barrier_household_chores = np.where(dropped_out, rng.binomial(1, 0.20), 0)

    # Transition (primary grade 6 → secondary grade 7)
    at_transition = (grade == 6).astype(int)
    transitioned = np.where(at_transition,
        rng.binomial(1, _logistic(0.55, ses + 0.2 * receives_scholarship + 0.1 * parent_values_girls_edu, 0.5)), 0)

    # Aspiration (1-5 scale)
    aspiration_score = np.clip(
        rng.normal(3 + 0.5 * ses + 0.3 * in_safe_spaces_programme + 0.2 * parent_values_girls_edu, 0.8, n),
        1, 5
    ).round(1)

    df = pd.DataFrame({
        "girl_id": ids, "country": countries, "district": districts, "urban": urban.astype(int),
        "age": age, "grade": grade, "wealth_quintile": wealth_quintile,
        "mother_education_years": mother_educ,
        "father_alive": father_alive, "mother_alive": mother_alive,
        "household_size": hh_size,
        "parent_values_girls_edu": parent_values_girls_edu,
        "receives_scholarship": receives_scholarship,
        "receives_school_meals": receives_school_meals,
        "in_safe_spaces_programme": in_safe_spaces_programme,
        "distance_to_school_km": np.round(distance_km, 1),
        "feels_safe_route_to_school": feels_safe_route,
        "experienced_harassment_route": experienced_harassment,
        "srgbv_experienced": srgbv_experienced,
        "has_menstruated": has_menstruated,
        "mhm_knowledge": mhm_knowledge,
        "has_sanitary_products": has_sanitary_products,
        "missed_school_menstruation": missed_school_menstruation,
        "enrolled": enrolled,
        "attendance_rate": np.round(attendance_rate, 3),
        "math_score": math_score,
        "literacy_score": literacy_score,
        "dropped_out": dropped_out,
        "barrier_marriage": barrier_marriage,
        "barrier_pregnancy": barrier_pregnancy,
        "barrier_cost": barrier_cost,
        "barrier_distance": barrier_distance,
        "barrier_household_chores": barrier_household_chores,
        "at_primary_secondary_transition": at_transition,
        "transitioned_to_secondary": transitioned,
        "aspiration_score": aspiration_score,
    })

    df = inject_missing(df,
        columns=["math_score", "literacy_score", "attendance_rate", "experienced_harassment_route"],
        rates=[0.05, 0.05, 0.03, 0.08],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
