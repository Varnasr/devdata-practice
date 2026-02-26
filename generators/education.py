"""
Generator 6: Education Outcomes
───────────────────────────────
Simulates a school-level + student-level dataset with test scores, attendance,
teacher characteristics, and school resources.

Structure: ~500 schools × 40-80 students each → ~30k students.

Realistic features:
  • Multi-level structure (students nested in schools)
  • School random effects (ICC ≈ 0.15-0.25)
  • Teacher quality affecting student outcomes
  • Gender gaps in STEM vs. reading
  • Distance to school → attendance → scores
  • Grade repetition and dropout
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing


def generate(n_schools: int = 500, seed: int = 202) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    rows = []
    student_counter = 1

    for s in range(n_schools):
        school_id = f"SCH-{s + 1:04d}"
        district = rng.choice([
            "Kilifi", "Nairobi", "Kisumu", "Dar es Salaam", "Kampala",
            "Addis Ababa", "Kigali", "Dhaka", "Kathmandu", "Accra",
            "Lagos", "Dakar", "Lilongwe", "Maputo", "Phnom Penh",
        ])
        urban = int(rng.random() < 0.35)
        school_type = rng.choice(["government", "private", "community"],
                                 p=[0.60, 0.20, 0.20])

        # School-level characteristics
        n_teachers = rng.integers(4, 25)
        pupil_teacher_ratio = rng.uniform(25, 80)
        pct_trained_teachers = np.clip(rng.beta(3, 2) * 100, 10, 100)
        has_library = int(rng.random() < (0.6 if school_type == "private" else 0.25))
        has_electricity = int(rng.random() < (0.7 if urban else 0.30))
        has_water = int(rng.random() < (0.6 if urban else 0.35))
        textbook_ratio = np.clip(rng.beta(2, 3) * 100, 5, 100)  # % students with textbooks

        # School-level quality (random effect)
        school_quality = rng.normal(0, 1)
        school_quality += 0.3 * urban + 0.4 * (school_type == "private")
        school_quality += 0.002 * pct_trained_teachers

        # Students per school
        n_students = rng.integers(35, 85)

        for _ in range(n_students):
            sid = f"STU-{student_counter:06d}"
            student_counter += 1

            grade = rng.choice([3, 4, 5, 6, 7, 8], p=[0.20, 0.20, 0.18, 0.16, 0.14, 0.12])
            female = int(rng.random() < 0.50)
            age = grade + 5 + rng.choice([0, 0, 0, 1, 1, 2], p=[0.40, 0.25, 0.15, 0.10, 0.06, 0.04])

            # Household SES (proxy)
            ses_score = rng.normal(0, 1) + 0.4 * urban + 0.5 * (school_type == "private")

            # Distance to school (km)
            distance = np.clip(rng.exponential(2.0 if urban else 4.5), 0.1, 15)

            # Attendance rate (affected by distance, gender, SES)
            attend_logit = 2.0 - 0.15 * distance + 0.3 * ses_score - 0.1 * (1 - female)
            attendance_rate = np.clip(1 / (1 + np.exp(-attend_logit)) + rng.normal(0, 0.05), 0.30, 1.0)

            # Meals at school
            school_meal = int(rng.random() < (0.45 if school_type == "government" else 0.20))

            # Test scores (0-100 scale)
            # Math
            math_score = (
                45
                + 8 * school_quality
                + 5 * ses_score
                + 15 * attendance_rate
                + 2 * (female * -1)  # slight gender gap in math
                + 1.5 * grade
                + 3 * has_library
                + rng.normal(0, 10)
            )
            math_score = np.clip(round(math_score), 0, 100)

            # Reading / language
            reading_score = (
                48
                + 7 * school_quality
                + 4.5 * ses_score
                + 14 * attendance_rate
                + 2 * female  # girls do better in reading
                + 1.5 * grade
                + 4 * has_library
                + rng.normal(0, 10)
            )
            reading_score = np.clip(round(reading_score), 0, 100)

            # Grade repetition
            repeated_grade = int(rng.random() < (0.20 - 0.03 * ses_score - 0.05 * attendance_rate))

            # Dropped out by endline
            dropout = int(rng.random() < np.clip(0.08 - 0.02 * ses_score + 0.01 * distance, 0.01, 0.25))

            rows.append({
                "student_id": sid,
                "school_id": school_id,
                "district": district,
                "urban": urban,
                "school_type": school_type,
                "n_teachers": n_teachers,
                "pupil_teacher_ratio": round(pupil_teacher_ratio, 1),
                "pct_trained_teachers": round(pct_trained_teachers, 1),
                "has_library": has_library,
                "has_electricity": has_electricity,
                "has_water": has_water,
                "textbook_student_ratio": round(textbook_ratio, 1),
                "grade": grade,
                "age": age,
                "female": female,
                "ses_score": round(ses_score, 2),
                "distance_to_school_km": round(distance, 1),
                "attendance_rate": round(attendance_rate, 3),
                "school_meal_program": school_meal,
                "math_score": math_score,
                "reading_score": reading_score,
                "repeated_grade": repeated_grade,
                "dropped_out": dropout,
            })

    df = pd.DataFrame(rows)

    # Dropout students have missing endline scores
    df.loc[df["dropped_out"] == 1, ["math_score", "reading_score"]] = np.nan

    df = inject_missing(df,
        columns=["ses_score", "attendance_rate", "distance_to_school_km"],
        rates=[0.04, 0.03, 0.05],
        rng=rng, mechanism="MCAR")
    return df
