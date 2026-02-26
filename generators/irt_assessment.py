"""
Generator: Psychometric / IRT Assessment
─────────────────────────────────────────
Simulates item-level response data from a psychometric assessment using
Item Response Theory (IRT) models. Each row is one respondent with
responses to 30 items, plus respondent demographics and test metadata.

Rows: ~20k respondents (one row per respondent, 30 item columns).

Realistic features:
  • 2PL and 3PL IRT models for response generation
  • Item difficulty (b) and discrimination (a) parameters
  • Guessing parameter (c) for multiple-choice items (3PL)
  • Latent ability (theta) driven by education, wealth, urban
  • Differential Item Functioning (DIF) by gender on select items
  • Response time modelled as lognormal (slower for harder items)
  • Total score and percent correct summaries
  • Missing data on response time columns (MCAR)
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_respondents: int = 20000, seed: int = 813) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_respondents
    n_items = 30

    ids = [f"RSP-{i:07d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.35)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.50, n)
    age = rng.integers(8, 55, n)
    educ_years = np.clip(rng.normal(7 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    wealth_quintile = np.clip(
        rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.21, 0.20, 0.19, 0.18])
        + np.where(urban, 1, 0),
        1, 5
    )

    test_type = rng.choice(
        ["literacy", "numeracy", "cognitive", "sel_competency"],
        n, p=[0.30, 0.30, 0.25, 0.15]
    )
    grade_level = np.clip(rng.integers(3, 12, n), 3, 12)

    # --- Latent ability (theta) ---
    theta = rng.normal(0, 1, n)
    theta += 0.05 * educ_years  # education boost
    theta += 0.15 * (wealth_quintile - 3) / 2  # wealth boost
    theta += 0.10 * urban.astype(float)  # urban boost

    # --- Item parameters (fixed across respondents) ---
    # Difficulty (b): standard normal-ish, range roughly -3 to 3
    item_b = np.sort(rng.uniform(-2.5, 2.5, n_items))  # sorted easy to hard
    # Discrimination (a): positive, 0.5 to 2.5
    item_a = rng.uniform(0.5, 2.5, n_items)
    # Guessing (c): ~0.25 for multiple choice items
    item_c = rng.uniform(0.15, 0.30, n_items)

    # --- Differential Item Functioning (DIF) ---
    # Items 5, 12, 18 favor females; Items 8, 22, 27 favor males
    dif_female_favored = [4, 11, 17]  # 0-indexed
    dif_male_favored = [7, 21, 26]

    # --- Generate item responses (one row per respondent, 30 item columns) ---
    responses = np.zeros((n, n_items), dtype=int)
    response_times = np.zeros((n, n_items))

    for j in range(n_items):
        # Effective theta with DIF adjustment
        theta_eff = theta.copy()
        if j in dif_female_favored:
            theta_eff += 0.4 * female - 0.2 * (1 - female)
        elif j in dif_male_favored:
            theta_eff += 0.4 * (1 - female) - 0.2 * female

        # 3PL model: P = c + (1-c) / (1 + exp(-a*(theta - b)))
        exponent = -item_a[j] * (theta_eff - item_b[j])
        prob = item_c[j] + (1 - item_c[j]) / (1 + np.exp(np.clip(exponent, -30, 30)))
        prob = np.clip(prob, 0, 1)
        responses[:, j] = rng.binomial(1, prob)

        # Response time: lognormal, harder items take longer
        # mean_log_time increases with item difficulty
        mean_log_time = 3.0 + 0.3 * item_b[j] - 0.1 * theta_eff
        response_times[:, j] = np.clip(
            rng.lognormal(mean_log_time, 0.5, n), 5, 600
        ).round(1)

    # --- Summaries ---
    total_score = responses.sum(axis=1)
    pct_correct = np.round(total_score / n_items * 100, 1)

    # Build DataFrame
    data = {
        "respondent_id": ids,
        "country": countries,
        "district": districts,
        "female": female,
        "age": age,
        "education_years": educ_years,
        "wealth_quintile": wealth_quintile,
        "test_type": test_type,
        "grade_level": grade_level,
        "theta": np.round(theta, 3),
    }

    # Item responses
    for j in range(n_items):
        data[f"item_{j+1}"] = responses[:, j]

    # Item parameters (repeated per row for analysis convenience)
    for j in range(n_items):
        data[f"item_{j+1}_difficulty"] = item_b[j]
        data[f"item_{j+1}_discrimination"] = round(item_a[j], 3)

    # Response times
    for j in range(n_items):
        data[f"item_{j+1}_response_time_sec"] = response_times[:, j]

    data["total_score"] = total_score
    data["pct_correct"] = pct_correct

    df = pd.DataFrame(data)

    # Inject missing on response time columns only
    rt_cols = [f"item_{j+1}_response_time_sec" for j in range(n_items)]
    rt_rates = [0.03] * n_items
    df = inject_missing(df,
        columns=rt_cols,
        rates=rt_rates,
        rng=rng, mechanism="MCAR")
    return df
