"""
Generator: Media & Information Ecosystems
──────────────────────────────────────────
Simulates individual-level data on media consumption, information access,
journalism engagement, media literacy, and development communication
across low- and middle-income countries.

Rows: ~18k individuals.

Realistic features:
  • Urban-rural divide in media access (urban = digital, rural = radio)
  • Youth more digital, elderly more radio/newspaper
  • Media literacy correlated with education
  • Misinformation exposure and sharing patterns
  • Development communication reach and effectiveness
  • Press freedom perceptions and self-censorship
  • Language barriers to information access
  • Programme effects on media literacy skills
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, individual_ids


def generate(n_individuals: int = 18000, seed: int = 812) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    ids = individual_ids(rng, n, prefix="MED")
    districts, urban = pick_districts(rng, n, urban_share=0.35)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.50, n)
    age = rng.integers(15, 75, n)
    youth = (age < 30).astype(float)
    elderly = (age >= 55).astype(float)
    educ_years = np.clip(rng.normal(7 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    wealth_quintile = np.clip(
        rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.21, 0.20, 0.19, 0.18])
        + np.where(urban, 1, 0),
        1, 5
    )
    wealth_z = (wealth_quintile - 3) / 2.0  # standardised wealth

    # --- Media access ---
    has_radio = rng.binomial(1, _logistic(0.70, -0.1 * urban.astype(float) + 0.1 * wealth_z, 0.3))
    has_tv = rng.binomial(1, _logistic(0.35, 0.3 * urban.astype(float) + 0.25 * wealth_z, 0.4))
    has_smartphone = rng.binomial(1, _logistic(0.30, 0.35 * urban.astype(float) + 0.3 * wealth_z + 0.3 * youth, 0.4))
    has_internet = rng.binomial(1, _logistic(0.20, 0.4 * urban.astype(float) + 0.25 * wealth_z + 0.25 * youth, 0.4))
    newspaper_access = rng.binomial(1, _logistic(0.15, 0.2 * urban.astype(float) + 0.15 * educ_years / 10, 0.3))

    # --- Consumption hours per week ---
    radio_hours = np.clip(
        rng.exponential(4 + 2 * elderly - 1.5 * youth + 1 * has_radio), 0, 40
    ).round(1)
    tv_hours = np.clip(
        rng.exponential(np.clip(3 + 3 * has_tv + 1 * urban.astype(float) - 1 * elderly, 0.5, None)), 0, 50
    ).round(1)
    social_media_hours = np.clip(
        rng.exponential(np.clip(2 + 4 * has_smartphone * has_internet + 3 * youth - 2 * elderly, 0.3, None)), 0, 50
    ).round(1)
    newspaper_reading = np.clip(
        rng.exponential(np.clip(0.5 + 1.5 * newspaper_access + 0.5 * educ_years / 10, 0.2, None)), 0, 15
    ).round(1)
    online_news_hours = np.clip(
        rng.exponential(np.clip(1 + 3 * has_internet + 2 * youth - 1.5 * elderly, 0.2, None)), 0, 30
    ).round(1)

    # --- Primary news source ---
    # Youth & urban lean toward social media; elderly & rural lean toward radio
    source_options = ["radio", "tv", "social_media", "newspaper", "community_leader", "word_of_mouth"]
    primary_news_source = []
    for i in range(n):
        if youth[i] and has_smartphone[i]:
            p = [0.10, 0.20, 0.40, 0.05, 0.05, 0.20]
        elif elderly[i]:
            p = [0.40, 0.20, 0.05, 0.10, 0.15, 0.10]
        elif urban[i]:
            p = [0.15, 0.30, 0.25, 0.10, 0.05, 0.15]
        else:
            p = [0.35, 0.15, 0.10, 0.05, 0.20, 0.15]
        primary_news_source.append(rng.choice(source_options, p=p))
    primary_news_source = np.array(primary_news_source)

    trusts_primary_source = np.clip(rng.normal(3.2, 0.9, n), 1, 5).round(0).astype(int)

    # --- Media literacy ---
    literacy_base = 0.3 + 0.03 * educ_years + 0.1 * youth + 0.05 * urban.astype(float)
    can_distinguish_news_opinion = rng.binomial(1, np.clip(literacy_base, 0.05, 0.95))
    can_identify_fake_news = rng.binomial(1, np.clip(literacy_base - 0.05, 0.05, 0.90))
    checks_multiple_sources = rng.binomial(1, np.clip(literacy_base - 0.10, 0.05, 0.90))
    media_literacy_score = np.clip(
        rng.normal(3.5 + 0.3 * educ_years / 3 + 0.5 * youth + 0.3 * urban.astype(float), 1.5, n),
        0, 10
    ).round(1)

    # --- Development communication ---
    exposed_to_dev_content = rng.binomial(1, _logistic(0.35, 0.1 * has_radio + 0.1 * has_tv + 0.05 * has_internet, 0.3))
    dev_content_topic = np.where(
        exposed_to_dev_content,
        rng.choice(
            ["health", "agriculture", "education", "rights", "governance", "climate"],
            n, p=[0.25, 0.20, 0.18, 0.12, 0.13, 0.12]
        ),
        "none"
    )
    dev_content_channel = np.where(
        exposed_to_dev_content,
        rng.choice(
            ["radio", "tv", "social_media", "community_meeting", "print", "sms"],
            n, p=[0.30, 0.22, 0.18, 0.15, 0.08, 0.07]
        ),
        "none"
    )
    found_dev_content_useful = np.where(
        exposed_to_dev_content,
        rng.binomial(1, 0.65),
        0
    )

    # --- Journalism ---
    community_reporter = rng.binomial(1, 0.03, n)
    citizen_journalist = rng.binomial(1, _logistic(0.02, 0.2 * has_smartphone * has_internet + 0.1 * youth, 0.4))
    reported_local_issue = rng.binomial(1, _logistic(0.08, 0.1 * educ_years / 10 + 0.05 * has_smartphone, 0.3))

    # --- Misinformation ---
    encountered_health_misinformation = rng.binomial(1, _logistic(0.30, 0.15 * social_media_hours / 10, 0.3))
    encountered_political_misinformation = rng.binomial(1, _logistic(0.25, 0.1 * social_media_hours / 10, 0.3))
    shared_misinformation = rng.binomial(1, np.clip(
        0.10 + 0.03 * social_media_hours / 10 - 0.02 * media_literacy_score / 10, 0.01, 0.50
    ))
    corrected_by_others = np.where(
        shared_misinformation,
        rng.binomial(1, 0.30),
        0
    )

    # --- Press freedom ---
    perceives_media_freedom = np.clip(rng.normal(2.8, 1.0, n), 1, 5).round(0).astype(int)
    self_censorship = rng.binomial(1, _logistic(0.25, -0.05 * perceives_media_freedom, 0.3))

    # --- Language ---
    primary_language = rng.choice(
        ["local_language", "national_language", "english", "french", "arabic", "other"],
        n, p=[0.35, 0.30, 0.12, 0.08, 0.07, 0.08]
    )
    content_available_in_language = rng.binomial(1, np.where(
        np.isin(primary_language, ["english", "french", "national_language"]), 0.80, 0.40
    ))
    language_barrier_to_info = rng.binomial(1, np.where(content_available_in_language, 0.10, 0.55))

    # --- Programme ---
    in_media_literacy_programme = rng.binomial(1, 0.12, n)
    programme_improved_skills = np.where(
        in_media_literacy_programme,
        rng.binomial(1, 0.70),
        0
    )

    df = pd.DataFrame({
        "individual_id": ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "female": female, "age": age,
        "education_years": educ_years, "wealth_quintile": wealth_quintile,
        "has_radio": has_radio, "has_tv": has_tv,
        "has_smartphone": has_smartphone, "has_internet": has_internet,
        "newspaper_access": newspaper_access,
        "radio_hours": radio_hours, "tv_hours": tv_hours,
        "social_media_hours": social_media_hours,
        "newspaper_reading": newspaper_reading,
        "online_news_hours": online_news_hours,
        "primary_news_source": primary_news_source,
        "trusts_primary_source": trusts_primary_source,
        "can_distinguish_news_opinion": can_distinguish_news_opinion,
        "can_identify_fake_news": can_identify_fake_news,
        "checks_multiple_sources": checks_multiple_sources,
        "media_literacy_score": media_literacy_score,
        "exposed_to_dev_content": exposed_to_dev_content,
        "dev_content_topic": dev_content_topic,
        "dev_content_channel": dev_content_channel,
        "found_dev_content_useful": found_dev_content_useful,
        "community_reporter": community_reporter,
        "citizen_journalist": citizen_journalist,
        "reported_local_issue": reported_local_issue,
        "encountered_health_misinformation": encountered_health_misinformation,
        "encountered_political_misinformation": encountered_political_misinformation,
        "shared_misinformation": shared_misinformation,
        "corrected_by_others": corrected_by_others,
        "perceives_media_freedom": perceives_media_freedom,
        "self_censorship": self_censorship,
        "primary_language": primary_language,
        "content_available_in_language": content_available_in_language,
        "language_barrier_to_info": language_barrier_to_info,
        "in_media_literacy_programme": in_media_literacy_programme,
        "programme_improved_skills": programme_improved_skills,
    })

    df = inject_missing(df,
        columns=["media_literacy_score", "radio_hours", "tv_hours",
                 "social_media_hours", "trusts_primary_source"],
        rates=[0.04, 0.05, 0.05, 0.06, 0.03],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
