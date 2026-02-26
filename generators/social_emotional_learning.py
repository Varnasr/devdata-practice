"""
Generator: Youth SEL Competencies
──────────────────────────────────
Simulates student-level social-emotional learning assessment data for
school-based programmes, covering the five CASEL SEL domains
(self-awareness, self-management, social awareness, relationship skills,
responsible decision-making), academic outcomes, wellbeing indicators,
behavioural measures, programme participation, teacher ratings, and
parent/home environment factors.

Rows: one per student (~15k), with cross-sectional assessment data.

Realistic features:
  • Five SEL domain scores (1-5) with a composite, correlated with
    programme participation, duration, and teacher training
  • Gender differences: girls score slightly higher on social awareness
    and relationship skills
  • Academic scores (math, reading) and attendance correlated with SEL
  • Wellbeing indicators: life satisfaction, safety, friendships, bullying
  • Behavioural outcomes: disciplinary incidents, prosocial behaviour
  • Programme effects: SEL programme improves all five domains, with
    bigger effects for longer programme duration
  • Teacher and parent engagement ratings
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_students: int = 15000, seed: int = 809) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_students

    # --- IDs & geography ---
    student_ids = [f"STU-{i:06d}" for i in range(1, n + 1)]
    countries = rng.choice(list(COUNTRIES.keys()), n)
    districts, urban = pick_districts(rng, n, urban_share=0.40)

    # --- Student demographics ---
    female = rng.binomial(1, 0.50, n)
    age = rng.integers(6, 19, n)  # 6-18 inclusive
    grade = np.clip(age - 5 + rng.integers(-1, 2, n), 1, 12)
    school_ids = rng.choice([f"SCH-{i:04d}" for i in range(1, 501)], n)

    # Latent ability (drives academic and SEL outcomes)
    ability_latent = rng.normal(0, 1, n) + 0.2 * urban.astype(float)

    # Wealth quintile
    wealth_latent = rng.normal(0, 1, n) + 0.3 * urban.astype(float)
    wealth_quintile = pd.qcut(wealth_latent, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # --- Parent / home environment ---
    parent_education_years = np.clip(
        rng.normal(7 + 2 * urban.astype(float), 4, n), 0, 20
    ).astype(int)
    parent_engagement_score = np.clip(
        rng.normal(3.0 + 0.2 * wealth_latent + 0.1 * parent_education_years / 10, 0.8, n),
        1, 5,
    ).round(1)
    home_learning_environment = np.clip(
        rng.normal(2.8 + 0.3 * wealth_latent + 0.15 * parent_education_years / 10, 0.9, n),
        1, 5,
    ).round(1)

    # --- Programme assignment ---
    in_sel_programme = rng.binomial(1, 0.45, n)
    programme_duration_months = np.where(
        in_sel_programme,
        rng.choice([3, 6, 9, 12, 18, 24], n, p=[0.10, 0.20, 0.25, 0.25, 0.12, 0.08]),
        0,
    )
    teacher_trained_in_sel = rng.binomial(
        1, _logistic(0.35, 0.4 * in_sel_programme.astype(float), 0.5)
    )

    # Programme effect: scales with duration (normalised to 0-1 range)
    prog_effect = in_sel_programme * (programme_duration_months / 24.0)

    # --- SEL domain scores (1-5 scale) ---
    # Base + ability + programme effect + teacher training + noise
    female_f = female.astype(float)

    self_awareness = np.clip(
        rng.normal(
            2.8 + 0.2 * ability_latent + 0.4 * prog_effect
            + 0.15 * teacher_trained_in_sel + 0.1 * parent_engagement_score / 5,
            0.7, n,
        ),
        1, 5,
    ).round(1)

    self_management = np.clip(
        rng.normal(
            2.7 + 0.2 * ability_latent + 0.35 * prog_effect
            + 0.15 * teacher_trained_in_sel + 0.05 * age / 18,
            0.7, n,
        ),
        1, 5,
    ).round(1)

    social_awareness = np.clip(
        rng.normal(
            2.9 + 0.15 * ability_latent + 0.45 * prog_effect
            + 0.15 * teacher_trained_in_sel + 0.12 * female_f,
            0.7, n,
        ),
        1, 5,
    ).round(1)

    relationship_skills = np.clip(
        rng.normal(
            2.85 + 0.15 * ability_latent + 0.4 * prog_effect
            + 0.12 * teacher_trained_in_sel + 0.10 * female_f,
            0.7, n,
        ),
        1, 5,
    ).round(1)

    responsible_decision_making = np.clip(
        rng.normal(
            2.6 + 0.2 * ability_latent + 0.38 * prog_effect
            + 0.15 * teacher_trained_in_sel + 0.08 * age / 18,
            0.75, n,
        ),
        1, 5,
    ).round(1)

    sel_composite_score = np.round(
        (self_awareness + self_management + social_awareness
         + relationship_skills + responsible_decision_making) / 5, 1
    )

    # --- Academic outcomes ---
    math_score = np.clip(
        rng.normal(
            50 + 8 * ability_latent + 3 * sel_composite_score
            + 2 * wealth_latent + 4 * urban.astype(float), 15, n,
        ),
        0, 100,
    ).round(1)

    reading_score = np.clip(
        rng.normal(
            52 + 7 * ability_latent + 3 * sel_composite_score
            + 2 * wealth_latent + 3 * urban.astype(float) + 2 * female_f, 14, n,
        ),
        0, 100,
    ).round(1)

    attendance_rate = np.clip(
        rng.normal(
            0.82 + 0.03 * sel_composite_score / 5 + 0.02 * wealth_latent
            + 0.02 * urban.astype(float), 0.10, n,
        ),
        0.20, 1.0,
    ).round(3)

    # --- Wellbeing ---
    life_satisfaction = np.clip(
        rng.normal(
            5.5 + 0.5 * sel_composite_score / 5 + 0.3 * wealth_latent
            + 0.2 * parent_engagement_score / 5, 1.8, n,
        ),
        1, 10,
    ).round(1)

    feels_safe_at_school = rng.binomial(
        1, _logistic(0.72, 0.15 * sel_composite_score / 5 + 0.1 * wealth_latent, 0.4)
    )
    has_close_friends = rng.binomial(
        1, _logistic(0.80, 0.1 * relationship_skills / 5 + 0.05 * social_awareness / 5, 0.3)
    )
    bullying_experienced = rng.binomial(
        1, _logistic(0.22, -0.15 * sel_composite_score / 5 - 0.1 * feels_safe_at_school, 0.4)
    )
    bullying_perpetrated = rng.binomial(
        1, _logistic(0.08, -0.12 * self_management / 5 - 0.1 * responsible_decision_making / 5, 0.4)
    )

    # --- Behaviour ---
    disciplinary_incidents = np.clip(
        rng.poisson(
            np.clip(1.5 - 0.3 * self_management - 0.2 * prog_effect, 0.1, 20), n
        ),
        0, 15,
    )

    prosocial_behavior_score = np.clip(
        rng.normal(
            5.0 + 0.8 * relationship_skills / 5 + 0.5 * social_awareness / 5
            + 0.3 * prog_effect, 1.5, n,
        ),
        0, 10,
    ).round(1)

    # --- Teacher ratings ---
    teacher_rated_behavior = np.clip(
        rng.normal(
            2.8 + 0.3 * self_management / 5 + 0.2 * prog_effect
            + 0.1 * teacher_trained_in_sel, 0.7, n,
        ),
        1, 5,
    ).round(1)

    teacher_rated_engagement = np.clip(
        rng.normal(
            2.9 + 0.25 * ability_latent * 0.3 + 0.2 * sel_composite_score / 5
            + 0.15 * prog_effect, 0.7, n,
        ),
        1, 5,
    ).round(1)

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "student_id": student_ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "female": female,
        "age": age,
        "grade": grade,
        "school_id": school_ids,
        "wealth_quintile": wealth_quintile,
        # SEL domains
        "self_awareness": self_awareness,
        "self_management": self_management,
        "social_awareness": social_awareness,
        "relationship_skills": relationship_skills,
        "responsible_decision_making": responsible_decision_making,
        "sel_composite_score": sel_composite_score,
        # Academic
        "math_score": math_score,
        "reading_score": reading_score,
        "attendance_rate": attendance_rate,
        # Wellbeing
        "life_satisfaction": life_satisfaction,
        "feels_safe_at_school": feels_safe_at_school,
        "has_close_friends": has_close_friends,
        "bullying_experienced": bullying_experienced,
        "bullying_perpetrated": bullying_perpetrated,
        # Behaviour
        "disciplinary_incidents": disciplinary_incidents,
        "prosocial_behavior_score": prosocial_behavior_score,
        # Programme
        "in_sel_programme": in_sel_programme,
        "programme_duration_months": programme_duration_months,
        "teacher_trained_in_sel": teacher_trained_in_sel,
        # Teacher ratings
        "teacher_rated_behavior": teacher_rated_behavior,
        "teacher_rated_engagement": teacher_rated_engagement,
        # Parent / home
        "parent_education_years": parent_education_years,
        "parent_engagement_score": parent_engagement_score,
        "home_learning_environment": home_learning_environment,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "self_awareness", "self_management", "social_awareness",
            "relationship_skills", "responsible_decision_making",
            "math_score", "reading_score", "life_satisfaction",
            "teacher_rated_behavior", "teacher_rated_engagement",
            "parent_engagement_score", "home_learning_environment",
        ],
        rates=[0.04, 0.04, 0.04, 0.04, 0.04,
               0.06, 0.06, 0.05,
               0.07, 0.07,
               0.08, 0.08],
        rng=rng,
        mechanism="MCAR",
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
