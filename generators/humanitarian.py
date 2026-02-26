"""
Generator: Humanitarian / Disaster Response
────────────────────────────────────────────
Simulates an emergency needs-assessment and response-monitoring dataset with
displacement tracking, multi-sector severity scoring, aid distribution,
protection concerns, communication with affected populations (CwC), and
accountability mechanisms — the kind produced by OCHA joint assessments,
UNHCR household surveys, and cluster-level monitoring.

Rows: one per individual/household (~18k), with SADD (Sex and Age
Disaggregated Data) throughout.

Realistic features:
  • Displacement status (IDP, refugee, returnee, host community) with
    duration and crisis typology
  • Multi-sector severity needs (food, shelter, health, WASH, protection,
    education) on a 1-5 scale following JIAF methodology
  • Sphere standards compliance indicators (water, shelter area, food kcal)
  • Aid distribution: modality (cash, voucher, in-kind), amount, timeliness,
    and beneficiary satisfaction
  • Protection concerns: GBV risk, child protection, HLP (housing, land,
    property), mine/UXO awareness
  • CwC: information access, preferred channels, feedback mechanisms
  • Accountability: complaints filed, resolution rates
  • Movement intentions (stay, return, relocate, seek asylum)
  • Vulnerability markers: unaccompanied minors, pregnant/lactating women,
    persons with disability, elderly living alone
"""

import numpy as np
import pandas as pd
from .utils import (
    pick_districts, inject_missing, COUNTRIES, household_ids, random_dates,
)


def generate(n_individuals: int = 18000, seed: int = 712) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    # --- IDs & geography ---
    ind_ids = [f"HUM-{i:06d}" for i in range(1, n + 1)]
    # Cluster individuals into households (~5 per HH)
    n_hh = n // 5
    hh_pool = household_ids(rng, n_hh, prefix="HHH")
    hh_assignment = rng.choice(hh_pool, n)

    districts, urban = pick_districts(rng, n, urban_share=0.40)
    # Focus on crisis-affected countries
    crisis_countries = [
        "Kenya", "Ethiopia", "Uganda", "Nigeria", "Bangladesh",
        "Pakistan", "Colombia", "Nepal", "Mozambique", "Cambodia",
    ]
    crisis_weights = np.array([0.12, 0.15, 0.10, 0.12, 0.12,
                               0.10, 0.08, 0.07, 0.07, 0.07])
    countries = rng.choice(crisis_countries, n, p=crisis_weights)

    # ------------------------------------------------------------------ #
    # SADD — Sex and Age Disaggregated Data
    # ------------------------------------------------------------------ #
    female = rng.binomial(1, 0.52, n)
    age = rng.integers(0, 85, n)
    # Ensure realistic age pyramid (more children in humanitarian settings)
    child_boost = rng.random(n) < 0.22
    age[child_boost] = rng.integers(0, 15, child_boost.sum())

    age_group = np.where(
        age < 5, "0-4",
        np.where(age < 12, "5-11",
        np.where(age < 18, "12-17",
        np.where(age < 60, "18-59", "60+")))
    )
    age_group_broad = np.where(age < 18, "child",
                      np.where(age < 60, "adult", "elderly"))

    hh_size = rng.choice(range(1, 13), n,
                         p=[0.02, 0.03, 0.06, 0.09, 0.13, 0.17,
                            0.16, 0.13, 0.09, 0.06, 0.04, 0.02])

    # ------------------------------------------------------------------ #
    # Displacement & crisis context
    # ------------------------------------------------------------------ #
    displacement_status = rng.choice(
        ["idp", "refugee", "returnee", "host_community"],
        n, p=[0.35, 0.20, 0.10, 0.35],
    )
    displaced = np.isin(displacement_status, ["idp", "refugee", "returnee"]).astype(int)

    months_displaced = np.where(
        displaced,
        np.clip(rng.exponential(14), 0.5, 72).round(0).astype(int),
        0,
    )
    times_displaced = np.where(
        displaced,
        np.clip(rng.poisson(1.5), 1, 6),
        0,
    )

    crisis_type = rng.choice(
        ["conflict", "flood", "drought", "earthquake", "epidemic",
         "mixed_conflict_natural", "volcano"],
        n, p=[0.28, 0.18, 0.14, 0.08, 0.12, 0.15, 0.05],
    )

    assessment_date = random_dates(rng, n, "2022-01-01", "2024-12-31")
    assessment_round = rng.choice([1, 2, 3, 4], n, p=[0.30, 0.30, 0.25, 0.15])

    # Current shelter type
    shelter_type = np.where(
        displaced,
        rng.choice(
            ["tent", "makeshift_shelter", "collective_centre",
             "host_family", "rented_accommodation", "transitional_shelter"],
            n, p=[0.20, 0.18, 0.15, 0.22, 0.15, 0.10]),
        rng.choice(
            ["permanent_house", "semi_permanent", "rented_accommodation",
             "makeshift_shelter", "other"],
            n, p=[0.40, 0.25, 0.20, 0.10, 0.05]),
    )

    # ------------------------------------------------------------------ #
    # Vulnerability markers
    # ------------------------------------------------------------------ #
    unaccompanied_minor = ((age < 18) & rng.binomial(1, 0.04, n).astype(bool)).astype(int)
    separated_child = ((age < 18) & (unaccompanied_minor == 0)
                       & rng.binomial(1, 0.06, n).astype(bool)).astype(int)
    pregnant_lactating = (female.astype(bool) & (age >= 15) & (age <= 49)
                          & rng.binomial(1, 0.12, n).astype(bool)).astype(int)
    has_disability = rng.binomial(1, 0.08, n)
    chronic_illness = rng.binomial(1, _logistic(0.10, 0.01 * np.maximum(age - 40, 0), 0.3))
    elderly_alone = ((age >= 60) & rng.binomial(1, 0.07, n).astype(bool)).astype(int)
    female_headed_hh = (female.astype(bool) & (age >= 18)
                        & rng.binomial(1, 0.30, n).astype(bool)).astype(int)

    vulnerability_score = (
        displaced
        + unaccompanied_minor
        + separated_child
        + pregnant_lactating
        + has_disability
        + chronic_illness
        + elderly_alone
        + female_headed_hh
        + (hh_size >= 8).astype(int)
    )

    # ------------------------------------------------------------------ #
    # Multi-sector needs assessment (severity 0-5, JIAF-like)
    # ------------------------------------------------------------------ #
    base_need = 1.2 + 1.0 * displaced + 0.25 * vulnerability_score + rng.normal(0, 0.3, n)

    need_food = np.clip(rng.normal(base_need + 0.5, 1.0, n), 0, 5).round(0).astype(int)
    need_shelter = np.clip(rng.normal(base_need + 0.2 * displaced, 1.2, n), 0, 5).round(0).astype(int)
    need_health = np.clip(rng.normal(base_need - 0.2 + 0.15 * chronic_illness
                                     + 0.10 * pregnant_lactating, 1.0, n), 0, 5).round(0).astype(int)
    need_wash = np.clip(rng.normal(base_need + 0.15, 1.1, n), 0, 5).round(0).astype(int)
    need_protection = np.clip(rng.normal(base_need - 0.3 + 0.25 * female
                                         + 0.30 * unaccompanied_minor, 1.2, n), 0, 5).round(0).astype(int)
    need_education = np.where(
        (age >= 5) & (age <= 17),
        np.clip(rng.normal(base_need + 0.1, 1.1, n), 0, 5).round(0).astype(int),
        0,
    )
    need_livelihoods = np.where(
        age >= 15,
        np.clip(rng.normal(base_need + 0.3, 1.0, n), 0, 5).round(0).astype(int),
        0,
    )

    overall_severity = np.clip(
        np.maximum.reduce([need_food, need_shelter, need_health,
                           need_wash, need_protection]),
        0, 5,
    )
    people_in_need = (overall_severity >= 3).astype(int)

    # ------------------------------------------------------------------ #
    # Sphere standards indicators
    # ------------------------------------------------------------------ #
    # Water: >= 15 L/person/day
    water_lpd = np.round(np.clip(
        rng.normal(14 - 3 * displaced + 4 * urban.astype(float), 6, n), 2, 80
    ), 1)
    meets_sphere_water = (water_lpd >= 15).astype(int)

    # Shelter: >= 3.5 m2 / person covered area
    shelter_area_m2pp = np.round(np.clip(
        rng.normal(np.where(displaced, 3.0, 5.5) + 1 * urban.astype(float), 1.5, n), 0.5, 20
    ), 1)
    meets_sphere_shelter = (shelter_area_m2pp >= 3.5).astype(int)

    # Food: >= 2,100 kcal / person / day
    kcal_per_person_day = np.round(np.clip(
        rng.normal(1900 - 200 * displaced + 100 * (1 - displaced), 300, n), 500, 3500
    ), 0).astype(int)
    meets_sphere_food = (kcal_per_person_day >= 2100).astype(int)

    # Food security scores
    fcs = np.round(np.clip(
        rng.normal(40 - 5 * displaced - 2.5 * vulnerability_score
                   + 3 * urban.astype(float), 12, n), 0, 112
    ), 1)
    fcs_category = np.where(fcs <= 21, "poor",
                   np.where(fcs <= 35, "borderline", "acceptable"))

    rcsi = np.clip(
        rng.normal(15 + 5 * displaced + 2 * vulnerability_score, 8, n),
        0, 56,
    ).round(0).astype(int)

    # ------------------------------------------------------------------ #
    # Aid distribution
    # ------------------------------------------------------------------ #
    received_aid = rng.binomial(1, _logistic(
        0.50, 0.20 * displaced + 0.10 * vulnerability_score, 0.3
    ))
    aid_modality = np.where(
        received_aid,
        rng.choice(["cash", "voucher", "in_kind", "mixed"],
                   n, p=[0.30, 0.18, 0.35, 0.17]),
        "none",
    )
    aid_amount_usd = np.where(
        received_aid,
        np.round(rng.lognormal(3.5, 0.6, n), 2),
        0.0,
    )
    aid_frequency = np.where(
        received_aid,
        rng.choice(["one_time", "monthly", "quarterly", "irregular"],
                   n, p=[0.25, 0.35, 0.20, 0.20]),
        "none",
    )
    aid_timeliness_score = np.where(
        received_aid,
        np.clip(rng.normal(3.0, 1.0, n), 1, 5).round(0).astype(int),
        0,
    )
    aid_sufficiency_score = np.where(
        received_aid,
        np.clip(rng.normal(2.8, 1.1, n), 1, 5).round(0).astype(int),
        0,
    )
    aid_satisfaction_score = np.where(
        received_aid,
        np.clip(rng.normal(3.2, 1.0, n), 1, 5).round(0).astype(int),
        0,
    )

    # ------------------------------------------------------------------ #
    # Protection concerns
    # ------------------------------------------------------------------ #
    feels_safe = rng.binomial(1, _logistic(
        0.55, -0.25 * displaced + 0.10 * urban.astype(float)
        - 0.05 * (crisis_type == "conflict").astype(float), 0.3
    ))

    # GBV
    gbv_risk_reported = rng.binomial(1, _logistic(
        0.18, 0.20 * female + 0.15 * displaced
        - 0.10 * feels_safe + 0.10 * (age >= 12).astype(float), 0.3
    ))
    gbv_services_available = rng.binomial(1, _logistic(
        0.35, 0.15 * urban.astype(float), 0.3
    ))
    gbv_services_accessed = np.where(
        gbv_risk_reported & gbv_services_available,
        rng.binomial(1, 0.30, n),
        0,
    )

    # Child protection
    child_protection_concern = np.where(
        age < 18,
        rng.binomial(1, _logistic(0.15, 0.20 * displaced
                                   + 0.30 * unaccompanied_minor
                                   + 0.15 * separated_child, 0.3)),
        0,
    )
    child_labor_reported = np.where(
        (age >= 5) & (age < 18),
        rng.binomial(1, _logistic(0.12, 0.15 * displaced
                                   - 0.05 * (need_education > 0).astype(float), 0.3)),
        0,
    )
    child_marriage_risk = np.where(
        (age >= 10) & (age < 18) & female.astype(bool),
        rng.binomial(1, _logistic(0.08, 0.10 * displaced, 0.3)),
        0,
    )

    # Mine / UXO awareness (in conflict/earthquake areas)
    conflict_mask = np.isin(crisis_type, ["conflict", "mixed_conflict_natural", "earthquake"])
    mine_uxo_awareness = np.where(
        conflict_mask,
        rng.binomial(1, _logistic(0.45, 0.10 * urban.astype(float), 0.2)),
        0,
    )
    mine_uxo_accident_reported = np.where(
        conflict_mask,
        rng.binomial(1, 0.04, n),
        0,
    )

    # HLP (Housing, Land, Property)
    hlp_concern = rng.binomial(1, _logistic(
        0.18, 0.20 * displaced + 0.10 * (crisis_type == "conflict").astype(float), 0.3
    ))
    has_documentation = rng.binomial(1, _logistic(
        0.52, -0.25 * displaced + 0.15 * urban.astype(float), 0.3
    ))

    # ------------------------------------------------------------------ #
    # Communication with Communities (CwC)
    # ------------------------------------------------------------------ #
    received_info_about_aid = rng.binomial(1, _logistic(
        0.42, 0.12 * received_aid + 0.08 * urban.astype(float), 0.3
    ))
    info_source = np.where(
        received_info_about_aid,
        rng.choice(
            ["community_leader", "radio", "phone_sms", "social_media",
             "ngo_staff", "loudspeaker", "notice_board", "word_of_mouth"],
            n, p=[0.18, 0.20, 0.14, 0.10, 0.12, 0.08, 0.05, 0.13]),
        "none",
    )
    preferred_info_channel = rng.choice(
        ["radio", "community_leader", "phone_sms", "social_media",
         "face_to_face", "loudspeaker", "written_notice"],
        n, p=[0.22, 0.18, 0.18, 0.14, 0.15, 0.08, 0.05],
    )
    language_barrier = rng.binomial(1, _logistic(
        0.12, 0.15 * (displacement_status == "refugee").astype(float), 0.3
    ))
    feels_informed = rng.binomial(1, _logistic(
        0.40, 0.15 * received_info_about_aid - 0.10 * language_barrier
        + 0.08 * urban.astype(float), 0.3
    ))

    # ------------------------------------------------------------------ #
    # Accountability mechanisms (feedback, complaints)
    # ------------------------------------------------------------------ #
    knows_feedback_mechanism = rng.binomial(1, _logistic(
        0.30, 0.12 * received_aid + 0.05 * received_info_about_aid, 0.3
    ))
    filed_complaint = np.where(
        knows_feedback_mechanism,
        rng.binomial(1, 0.18, n),
        0,
    )
    complaint_type = np.where(
        filed_complaint,
        rng.choice(["aid_exclusion", "aid_quality", "staff_behaviour",
                     "targeting_fairness", "delay", "other"],
                   n, p=[0.22, 0.18, 0.12, 0.20, 0.18, 0.10]),
        "none",
    )
    complaint_resolved = np.where(
        filed_complaint.astype(bool),
        rng.binomial(1, 0.45, n),
        0,
    )
    complaint_resolution_days = np.where(
        complaint_resolved,
        np.clip(rng.exponential(12), 1, 60).astype(int),
        0,
    )

    # ------------------------------------------------------------------ #
    # Movement intentions
    # ------------------------------------------------------------------ #
    movement_intention = np.where(
        displaced,
        rng.choice(
            ["stay_current", "return_home", "relocate_within_country",
             "seek_asylum_abroad", "undecided"],
            n, p=[0.30, 0.25, 0.15, 0.10, 0.20]),
        rng.choice(
            ["stay_current", "relocate_within_country", "undecided"],
            n, p=[0.70, 0.15, 0.15]),
    )

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "individual_id": ind_ids,
        "household_id": hh_assignment,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        # SADD
        "female": female,
        "age": age,
        "age_group": age_group,
        "age_group_broad": age_group_broad,
        "household_size": hh_size,
        # Displacement
        "displacement_status": displacement_status,
        "displaced": displaced,
        "months_displaced": months_displaced,
        "times_displaced": times_displaced,
        "crisis_type": crisis_type,
        "assessment_date": assessment_date,
        "assessment_round": assessment_round,
        "shelter_type": shelter_type,
        # Vulnerability
        "unaccompanied_minor": unaccompanied_minor,
        "separated_child": separated_child,
        "pregnant_lactating": pregnant_lactating,
        "has_disability": has_disability,
        "chronic_illness": chronic_illness,
        "elderly_alone": elderly_alone,
        "female_headed_hh": female_headed_hh,
        "vulnerability_score": vulnerability_score,
        # Needs
        "need_food": need_food,
        "need_shelter": need_shelter,
        "need_health": need_health,
        "need_wash": need_wash,
        "need_protection": need_protection,
        "need_education": need_education,
        "need_livelihoods": need_livelihoods,
        "overall_severity": overall_severity,
        "people_in_need": people_in_need,
        # Sphere standards
        "water_liters_per_person_day": water_lpd,
        "meets_sphere_water": meets_sphere_water,
        "shelter_area_m2_per_person": shelter_area_m2pp,
        "meets_sphere_shelter": meets_sphere_shelter,
        "kcal_per_person_day": kcal_per_person_day,
        "meets_sphere_food": meets_sphere_food,
        # Food security
        "food_consumption_score": fcs,
        "fcs_category": fcs_category,
        "reduced_coping_strategy_index": rcsi,
        # Aid
        "received_aid": received_aid,
        "aid_modality": aid_modality,
        "aid_amount_usd": aid_amount_usd,
        "aid_frequency": aid_frequency,
        "aid_timeliness_score": aid_timeliness_score,
        "aid_sufficiency_score": aid_sufficiency_score,
        "aid_satisfaction_score": aid_satisfaction_score,
        # Protection
        "feels_safe": feels_safe,
        "gbv_risk_reported": gbv_risk_reported,
        "gbv_services_available": gbv_services_available,
        "gbv_services_accessed": gbv_services_accessed,
        "child_protection_concern": child_protection_concern,
        "child_labor_reported": child_labor_reported,
        "child_marriage_risk": child_marriage_risk,
        "mine_uxo_awareness": mine_uxo_awareness,
        "mine_uxo_accident_reported": mine_uxo_accident_reported,
        "hlp_concern": hlp_concern,
        "has_documentation": has_documentation,
        # CwC
        "received_info_about_aid": received_info_about_aid,
        "info_source": info_source,
        "preferred_info_channel": preferred_info_channel,
        "language_barrier": language_barrier,
        "feels_informed": feels_informed,
        # Accountability
        "knows_feedback_mechanism": knows_feedback_mechanism,
        "filed_complaint": filed_complaint,
        "complaint_type": complaint_type,
        "complaint_resolved": complaint_resolved,
        "complaint_resolution_days": complaint_resolution_days,
        # Movement
        "movement_intention": movement_intention,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "food_consumption_score", "aid_amount_usd", "vulnerability_score",
            "gbv_risk_reported", "kcal_per_person_day", "shelter_area_m2_per_person",
            "water_liters_per_person_day", "complaint_resolution_days",
        ],
        rates=[0.05, 0.04, 0.03, 0.10, 0.06, 0.07, 0.05, 0.08],
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
