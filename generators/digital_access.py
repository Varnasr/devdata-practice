"""
Generator: Digital Divide & Data Literacy
──────────────────────────────────────────
Simulates individual-level data on digital access, literacy, privacy,
misinformation, and the gender/age digital divide.

Rows: ~20k individuals.

Realistic features:
  • Device ownership: mobile phone, smartphone, computer
  • Connectivity: internet access type, frequency, cost
  • Digital literacy: hierarchical skills from basic calls to online banking
  • Usage: social media, mobile money, e-government, telemedicine
  • Information ecosystem: misinformation encounter and identification
  • Privacy: data awareness, online harassment, data breach
  • Barriers to access: cost, literacy, language, relevance, infrastructure
  • Gender digital divide: women have lower access and literacy
  • Age divide: younger cohorts more digitally literate
  • Digital training programme and outcomes
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, individual_ids


def generate(n_individuals: int = 20000, seed: int = 808) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    ids = individual_ids(rng, n, prefix="DIG")
    districts, urban = pick_districts(rng, n, urban_share=0.38)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.51, n)
    age = rng.integers(15, 70, n)
    educ_years = np.clip(rng.normal(6 + 2 * urban.astype(float) - 0.3 * female, 3.5, n), 0, 18).astype(int)
    wealth_quintile = rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.21, 0.20, 0.19, 0.18])
    wealth = (wealth_quintile - 3) / 2.0 + rng.normal(0, 0.3, n)

    # Age effect: younger = more digital
    young = (age < 30).astype(float)
    middle = ((age >= 30) & (age < 50)).astype(float)
    # old implied by neither young nor middle

    # --- Device access ---
    # Gender digital divide: women have lower access
    owns_mobile_phone = rng.binomial(
        1, _logistic(0.70, wealth * 0.3 + 0.15 * urban.astype(float) - 0.15 * female, 0.4)
    )
    owns_smartphone = rng.binomial(
        1, _logistic(0.35, wealth * 0.4 + 0.2 * urban.astype(float) + 0.15 * young
                     - 0.15 * female + 0.1 * educ_years / 10, 0.5)
    )
    owns_computer = rng.binomial(
        1, _logistic(0.10, wealth * 0.5 + 0.2 * urban.astype(float) + 0.1 * educ_years / 10
                     - 0.08 * female, 0.5)
    )
    shared_device_only = ((~owns_mobile_phone.astype(bool)) & (~owns_smartphone.astype(bool))
                          & (~owns_computer.astype(bool))).astype(int)
    # Some shared-device-only people do have shared access
    shared_device_only = np.where(
        shared_device_only,
        rng.binomial(1, 0.45, n),
        0
    )

    # --- Connectivity ---
    has_internet_access = rng.binomial(
        1, _logistic(0.40, wealth * 0.3 + 0.25 * urban.astype(float) + 0.15 * young
                     + 0.1 * owns_smartphone - 0.12 * female, 0.4)
    )
    internet_type_options = ["mobile_data", "wifi", "broadband", "none"]
    internet_type = np.where(
        has_internet_access,
        rng.choice(["mobile_data", "wifi", "broadband"],
                   n, p=[0.55, 0.30, 0.15]),
        "none"
    )
    internet_freq_options = ["daily", "weekly", "monthly", "rarely", "never"]
    internet_frequency = np.where(
        has_internet_access,
        np.where(young.astype(bool),
                 rng.choice(["daily", "weekly", "monthly", "rarely"],
                            n, p=[0.55, 0.30, 0.10, 0.05]),
                 rng.choice(["daily", "weekly", "monthly", "rarely"],
                            n, p=[0.30, 0.35, 0.20, 0.15])),
        "never"
    )
    monthly_data_cost_usd = np.where(
        has_internet_access,
        np.clip(rng.lognormal(np.log(np.clip(3 + 2 * wealth, 0.1, None)), 0.5, n), 0.5, 50),
        0
    ).round(2)

    # --- Digital literacy (hierarchical: basic to advanced) ---
    # Gender divide and age divide built in
    base_lit = (0.15 * educ_years / 10 + 0.2 * young + 0.1 * middle
                + 0.15 * wealth + 0.1 * urban.astype(float) - 0.15 * female)
    can_make_call = rng.binomial(1, _logistic(0.80, base_lit, 0.3))
    can_send_sms = rng.binomial(1, _logistic(0.65, base_lit + 0.05 * can_make_call, 0.3))
    can_use_internet = rng.binomial(1, _logistic(0.40, base_lit + 0.1 * has_internet_access, 0.4))
    can_use_email = rng.binomial(1, _logistic(0.25, base_lit + 0.1 * can_use_internet, 0.4))
    can_use_social_media = rng.binomial(1, _logistic(0.35, base_lit + 0.15 * can_use_internet + 0.1 * young, 0.4))
    can_do_online_banking = rng.binomial(1, _logistic(0.15, base_lit + 0.1 * can_use_internet + 0.1 * wealth, 0.5))
    can_use_govt_services_online = rng.binomial(1, _logistic(0.12, base_lit + 0.1 * can_use_internet + 0.05 * educ_years / 10, 0.5))

    digital_literacy_score = np.clip(
        (can_make_call + can_send_sms + can_use_internet + can_use_email
         + can_use_social_media + can_do_online_banking + can_use_govt_services_online)
        / 7 * 10 + rng.normal(0, 0.3, n), 0, 10
    ).round(1)

    # --- Usage ---
    uses_social_media = rng.binomial(1, _logistic(0.30, base_lit + 0.15 * has_internet_access + 0.1 * young, 0.4))
    uses_mobile_money = rng.binomial(1, _logistic(0.25, wealth * 0.2 + 0.1 * owns_mobile_phone + 0.1 * urban.astype(float), 0.4))
    uses_e_government = rng.binomial(1, _logistic(0.10, base_lit + 0.1 * can_use_govt_services_online + 0.05 * educ_years / 10, 0.4))
    uses_online_education = rng.binomial(1, _logistic(0.12, base_lit + 0.1 * has_internet_access + 0.1 * young, 0.4))
    uses_telemedicine = rng.binomial(1, _logistic(0.08, base_lit + 0.1 * has_internet_access, 0.4))

    # --- Information ecosystem ---
    primary_info_source = rng.choice(
        ["radio", "tv", "newspaper", "social_media", "word_of_mouth", "internet"],
        n, p=[0.22, 0.20, 0.08, 0.18, 0.17, 0.15]
    )
    # Younger / more connected more likely to encounter misinformation
    encountered_misinformation = rng.binomial(
        1, _logistic(0.30, 0.15 * uses_social_media + 0.1 * has_internet_access + 0.05 * young, 0.4)
    )
    can_identify_misinformation = rng.binomial(
        1, _logistic(0.35, 0.1 * educ_years / 10 + 0.1 * digital_literacy_score / 10, 0.4)
    )
    shared_unverified_info = rng.binomial(
        1, _logistic(0.15, 0.1 * uses_social_media - 0.1 * can_identify_misinformation, 0.3)
    )

    # --- Privacy ---
    aware_of_data_privacy = rng.binomial(
        1, _logistic(0.20, 0.1 * educ_years / 10 + 0.1 * digital_literacy_score / 10 + 0.05 * urban.astype(float), 0.4)
    )
    experienced_online_harassment = rng.binomial(
        1, _logistic(0.10, 0.1 * uses_social_media + 0.08 * female + 0.05 * young, 0.4)
    )
    experienced_data_breach = rng.binomial(
        1, _logistic(0.06, 0.1 * has_internet_access + 0.05 * uses_mobile_money, 0.3)
    )
    uses_privacy_settings = rng.binomial(
        1, _logistic(0.18, 0.15 * aware_of_data_privacy + 0.1 * digital_literacy_score / 10, 0.4)
    )

    # --- Barriers ---
    no_access = (~has_internet_access.astype(bool)).astype(float)
    barrier_cost = rng.binomial(1, _logistic(0.35, no_access * 0.3 - wealth * 0.2, 0.3))
    barrier_literacy = rng.binomial(1, _logistic(0.25, no_access * 0.2 - 0.1 * educ_years / 10 + 0.1 * female, 0.3))
    barrier_language = rng.binomial(1, _logistic(0.18, no_access * 0.15 - 0.05 * educ_years / 10, 0.3))
    barrier_relevance = rng.binomial(1, _logistic(0.15, no_access * 0.2 - 0.05 * young, 0.3))
    barrier_infrastructure = rng.binomial(
        1, _logistic(0.25, no_access * 0.3 - 0.2 * urban.astype(float), 0.3)
    )

    # --- Programme ---
    received_digital_training = rng.binomial(1, 0.15, n)
    training_improved_skills = np.where(
        received_digital_training,
        rng.binomial(1, 0.65 + 0.10 * young),
        0
    )

    df = pd.DataFrame({
        "individual_id": ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "female": female, "age": age,
        "education_years": educ_years, "wealth_quintile": wealth_quintile,
        "owns_mobile_phone": owns_mobile_phone,
        "owns_smartphone": owns_smartphone,
        "owns_computer": owns_computer,
        "shared_device_only": shared_device_only,
        "has_internet_access": has_internet_access,
        "internet_type": internet_type,
        "internet_frequency": internet_frequency,
        "monthly_data_cost_usd": monthly_data_cost_usd,
        "can_make_call": can_make_call,
        "can_send_sms": can_send_sms,
        "can_use_internet": can_use_internet,
        "can_use_email": can_use_email,
        "can_use_social_media": can_use_social_media,
        "can_do_online_banking": can_do_online_banking,
        "can_use_govt_services_online": can_use_govt_services_online,
        "digital_literacy_score": digital_literacy_score,
        "uses_social_media": uses_social_media,
        "uses_mobile_money": uses_mobile_money,
        "uses_e_government": uses_e_government,
        "uses_online_education": uses_online_education,
        "uses_telemedicine": uses_telemedicine,
        "primary_info_source": primary_info_source,
        "encountered_misinformation": encountered_misinformation,
        "can_identify_misinformation": can_identify_misinformation,
        "shared_unverified_info": shared_unverified_info,
        "aware_of_data_privacy": aware_of_data_privacy,
        "experienced_online_harassment": experienced_online_harassment,
        "experienced_data_breach": experienced_data_breach,
        "uses_privacy_settings": uses_privacy_settings,
        "barrier_cost": barrier_cost,
        "barrier_literacy": barrier_literacy,
        "barrier_language": barrier_language,
        "barrier_relevance": barrier_relevance,
        "barrier_infrastructure": barrier_infrastructure,
        "received_digital_training": received_digital_training,
        "training_improved_skills": training_improved_skills,
    })

    df = inject_missing(df,
        columns=["monthly_data_cost_usd", "digital_literacy_score", "education_years", "internet_frequency"],
        rates=[0.06, 0.04, 0.03, 0.05],
        rng=rng, mechanism="MAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
