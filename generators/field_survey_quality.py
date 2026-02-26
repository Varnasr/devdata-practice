"""
Generator: Survey Paradata & Quality Control
─────────────────────────────────────────────
Simulates survey process data (paradata) for quality assurance training,
including interview timing, GPS validation, response pattern analysis,
back-check verification, and enumerator performance monitoring.

Rows: ~20k survey records from ~200 enumerators.

Realistic features:
  • Interview duration distributions with too-short / too-long flags
  • GPS accuracy and cluster distance validation
  • Straightlining, acquiescence, and digit preference detection
  • Back-check verification with match rates
  • Enumerator fatigue effects (surveys per day)
  • Fabrication patterns (short duration + straightlining + GPS issues)
  • Supervisor spot-check and audio audit data
  • Missing data rates and outlier counts
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_surveys: int = 20000, seed: int = 814) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_surveys

    survey_ids = [f"SRV-{i:06d}" for i in range(1, n + 1)]
    respondent_ids = [f"RES-{i:06d}" for i in range(1, n + 1)]

    # --- 200 enumerators ---
    n_enumerators = 200
    enum_ids_pool = [f"ENUM-{i:04d}" for i in range(1, n_enumerators + 1)]
    enumerator_id = rng.choice(enum_ids_pool, n)

    # Enumerator-level attributes (fixed per enumerator)
    enum_experience = {}
    enum_female = {}
    enum_education = {}
    enum_is_fabricator = {}  # ~5% are problematic
    for eid in enum_ids_pool:
        enum_experience[eid] = int(np.clip(rng.exponential(12), 1, 60))
        enum_female[eid] = int(rng.random() < 0.35)
        enum_education[eid] = rng.choice(["secondary", "diploma", "bachelors", "masters"],
                                          p=[0.25, 0.30, 0.35, 0.10])
        enum_is_fabricator[eid] = int(rng.random() < 0.05)

    enumerator_experience_months = np.array([enum_experience[e] for e in enumerator_id])
    enumerator_female = np.array([enum_female[e] for e in enumerator_id])
    enumerator_education = np.array([enum_education[e] for e in enumerator_id])
    is_fabricator = np.array([enum_is_fabricator[e] for e in enumerator_id])

    districts, urban = pick_districts(rng, n, urban_share=0.35)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # --- Supervisor ---
    n_supervisors = 25
    sup_ids_pool = [f"SUP-{i:03d}" for i in range(1, n_supervisors + 1)]
    supervisor_id = rng.choice(sup_ids_pool, n)

    # --- Timing ---
    # Normal interview: 30-50 min; fabricators: often very short
    base_duration = np.where(
        is_fabricator,
        rng.lognormal(2.3, 0.6, n),   # fabricators: median ~10 min
        rng.lognormal(3.5, 0.35, n)    # normal: median ~33 min
    )
    interview_duration_min = np.clip(base_duration, 3, 180).round(1)

    # Working hours: 7am to 7pm normal
    # Generate start hour
    start_hour = np.where(
        is_fabricator,
        rng.integers(5, 23, n),  # fabricators work odd hours
        np.clip(rng.normal(10, 2, n), 7, 18).astype(int)
    )
    start_minute = rng.integers(0, 60, n)
    interview_start_time = [f"{int(h):02d}:{int(m):02d}" for h, m in zip(start_hour, start_minute)]

    end_minutes = start_hour * 60 + start_minute + interview_duration_min
    end_hour = np.clip((end_minutes / 60).astype(int), 0, 23)
    end_min = (end_minutes % 60).astype(int)
    interview_end_time = [f"{int(h):02d}:{int(m):02d}" for h, m in zip(end_hour, end_min)]

    travel_time_min = np.clip(
        rng.exponential(np.where(urban, 15, 35)), 2, 180
    ).round(1)

    # Quality flags
    too_short = (interview_duration_min < 15).astype(int)
    too_long = (interview_duration_min > 90).astype(int)
    outside_working_hours = ((start_hour < 7) | (start_hour >= 19)).astype(int)
    day_of_week = rng.integers(0, 7, n)  # 0=Mon, 6=Sun
    weekend_interview = (day_of_week >= 5).astype(int)

    # --- GPS ---
    # Base coordinates (roughly East Africa)
    gps_latitude = rng.normal(-1.5, 3.0, n).round(6)
    gps_longitude = rng.normal(35.0, 5.0, n).round(6)
    gps_accuracy_m = np.clip(rng.exponential(8, n), 1, 200).round(1)

    # Distance from cluster center: fabricators often far away
    distance_from_cluster_center_km = np.where(
        is_fabricator,
        np.clip(rng.exponential(8, n), 0.1, 30),
        np.clip(rng.exponential(1.2, n), 0.01, 10)
    ).round(2)
    gps_suspicious = (distance_from_cluster_center_km > 5).astype(int)

    # --- Consent ---
    consent_obtained = rng.binomial(1, np.where(is_fabricator, 0.99, 0.92), n)
    respondent_available = rng.binomial(1, 0.85, n)
    callback_needed = rng.binomial(1, np.where(respondent_available, 0.05, 0.60))
    n_visit_attempts = np.where(
        respondent_available,
        rng.choice([1, 1, 1, 2, 2, 3], n),
        rng.choice([1, 2, 2, 3, 3, 3], n)
    )

    # --- Response patterns ---
    # Straightlining: fabricators have high scores
    straightlining_score = np.where(
        is_fabricator,
        np.clip(rng.beta(5, 2, n), 0.3, 1.0),
        np.clip(rng.beta(2, 8, n), 0, 0.5)
    ).round(3)

    # Acquiescence bias
    acquiescence_score = np.where(
        is_fabricator,
        np.clip(rng.beta(5, 2, n), 0.4, 1.0),
        np.clip(rng.beta(3, 4, n), 0, 1.0)
    ).round(3)

    # Digit preference (terminal digit preference in numeric answers)
    digit_preference_score = np.where(
        is_fabricator,
        np.clip(rng.beta(4, 2, n), 0.2, 1.0),
        np.clip(rng.beta(2, 6, n), 0, 0.6)
    ).round(3)

    # --- Back-check ---
    back_checked = rng.binomial(1, 0.15, n)
    back_check_match_rate = np.where(
        back_checked,
        np.where(
            is_fabricator,
            np.clip(rng.beta(3, 8, n), 0.1, 0.8),  # fabricators: low match
            np.clip(rng.beta(8, 2, n), 0.5, 1.0)    # normal: high match
        ),
        np.nan
    ).round(3) if True else None
    # Compute properly
    _match_raw = np.where(
        is_fabricator,
        np.clip(rng.beta(3, 8, n), 0.1, 0.8),
        np.clip(rng.beta(8, 2, n), 0.5, 1.0)
    )
    back_check_match_rate = np.where(back_checked, np.round(_match_raw, 3), np.nan)

    key_variable_discrepancy = np.where(
        back_checked,
        rng.binomial(1, np.where(is_fabricator, 0.60, 0.08)),
        0
    )

    # --- Enumerator workload ---
    # Surveys completed today (realistic: 3-8 typical, fabricators do more)
    surveys_completed_today = np.where(
        is_fabricator,
        rng.integers(6, 15, n),
        np.clip(rng.poisson(np.clip(4 + 1 * urban.astype(float), 1, None), n), 1, 12)
    )
    fatigue_flag = (surveys_completed_today > 6).astype(int)

    # --- Supervisor checks ---
    field_spot_checked = rng.binomial(1, 0.10, n)
    audio_recorded = rng.binomial(1, 0.25, n)
    audio_reviewed = np.where(audio_recorded, rng.binomial(1, 0.40), 0)

    # --- Data quality metrics ---
    missing_rate = np.where(
        is_fabricator,
        np.clip(rng.beta(1, 15, n), 0, 0.3),  # fabricators: suspiciously low missing
        np.clip(rng.beta(2, 20, n), 0, 0.4)
    ).round(3)

    dont_know_rate = np.where(
        is_fabricator,
        np.clip(rng.beta(1, 30, n), 0, 0.15),  # fabricators: suspiciously low
        np.clip(rng.beta(2, 15, n), 0, 0.3)
    ).round(3)

    refused_rate = np.clip(rng.beta(1, 40, n), 0, 0.15).round(3)

    outlier_count = np.clip(
        rng.poisson(np.clip(1.5 + 2 * is_fabricator, 0.1, None), n), 0, 20
    )

    df = pd.DataFrame({
        "survey_id": survey_ids,
        "enumerator_id": enumerator_id,
        "country": countries,
        "district": districts,
        "respondent_id": respondent_ids,
        "interview_start_time": interview_start_time,
        "interview_end_time": interview_end_time,
        "interview_duration_min": interview_duration_min,
        "travel_time_min": travel_time_min,
        "too_short": too_short,
        "too_long": too_long,
        "outside_working_hours": outside_working_hours,
        "weekend_interview": weekend_interview,
        "gps_latitude": gps_latitude,
        "gps_longitude": gps_longitude,
        "gps_accuracy_m": gps_accuracy_m,
        "distance_from_cluster_center_km": distance_from_cluster_center_km,
        "gps_suspicious": gps_suspicious,
        "consent_obtained": consent_obtained,
        "respondent_available": respondent_available,
        "callback_needed": callback_needed,
        "n_visit_attempts": n_visit_attempts,
        "straightlining_score": straightlining_score,
        "acquiescence_score": acquiescence_score,
        "digit_preference_score": digit_preference_score,
        "back_checked": back_checked,
        "back_check_match_rate": back_check_match_rate,
        "key_variable_discrepancy": key_variable_discrepancy,
        "enumerator_experience_months": enumerator_experience_months,
        "enumerator_female": enumerator_female,
        "enumerator_education": enumerator_education,
        "surveys_completed_today": surveys_completed_today,
        "fatigue_flag": fatigue_flag,
        "supervisor_id": supervisor_id,
        "field_spot_checked": field_spot_checked,
        "audio_recorded": audio_recorded,
        "audio_reviewed": audio_reviewed,
        "missing_rate": missing_rate,
        "dont_know_rate": dont_know_rate,
        "refused_rate": refused_rate,
        "outlier_count": outlier_count,
    })

    df = inject_missing(df,
        columns=["interview_duration_min", "travel_time_min", "gps_accuracy_m",
                 "back_check_match_rate", "straightlining_score"],
        rates=[0.03, 0.05, 0.04, 0.06, 0.03],
        rng=rng, mechanism="MCAR")
    return df
