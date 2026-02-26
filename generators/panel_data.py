"""
Generator 3: Country-Level Panel Data (WDI-style)
──────────────────────────────────────────────────
Simulates a balanced panel of 25 developing countries × 25 years (2000-2024)
with 15+ development indicators that co-move realistically.

Realistic features:
  • Indicators follow country-specific trends with AR(1) persistence
  • Cross-indicator correlations (GDP ↔ life expectancy, education ↔ fertility)
  • Structural breaks (e.g., COVID-2020 GDP shock)
  • Realistic ranges calibrated to actual WDI values
  • MCAR missingness at ~3-5% mimicking WDI gaps
"""

import numpy as np
import pandas as pd
from .utils import COUNTRIES


def generate(seed: int = 314) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    countries = list(COUNTRIES.keys())
    years = list(range(2000, 2025))
    n_c = len(countries)
    n_t = len(years)

    rows = []
    for ci, country in enumerate(countries):
        meta = COUNTRIES[country]
        region = meta["region"]
        base_gdppc = meta["gdppc"]

        # Country-specific growth rate
        trend_growth = rng.uniform(0.02, 0.06)

        # AR(1) shock process
        shock = np.zeros(n_t)
        for t in range(1, n_t):
            shock[t] = 0.6 * shock[t - 1] + rng.normal(0, 0.02)

        for ti, year in enumerate(years):
            elapsed = ti

            # GDP per capita (USD current)
            covid_shock = -0.08 if year == 2020 else (-0.02 if year == 2021 else 0)
            gdppc = base_gdppc * (1 + trend_growth) ** elapsed * np.exp(shock[ti] + covid_shock)

            # Log GDP (useful for regressions)
            log_gdppc = np.log(gdppc)

            # Life expectancy (rises with GDP, slowly)
            base_le = 52 + 5 * (log_gdppc - 6) + rng.normal(0, 0.5)
            le = np.clip(base_le + 0.15 * elapsed, 45, 80)

            # Infant mortality (per 1000, inversely related)
            imr = np.clip(120 - 12 * (log_gdppc - 5.5) - 1.5 * elapsed + rng.normal(0, 4), 5, 150)

            # Under-5 mortality
            u5mr = np.clip(imr * rng.uniform(1.3, 1.6) + rng.normal(0, 3), 8, 250)

            # Primary enrollment rate
            primary_enroll = np.clip(
                65 + 4 * (log_gdppc - 6) + 0.8 * elapsed + rng.normal(0, 3), 40, 100
            )

            # Secondary enrollment rate (lower)
            secondary_enroll = np.clip(
                primary_enroll * rng.uniform(0.45, 0.75) + rng.normal(0, 3), 15, 100
            )

            # Literacy rate (adult)
            literacy = np.clip(
                50 + 6 * (log_gdppc - 5.5) + 0.5 * elapsed + rng.normal(0, 2), 25, 99
            )

            # Fertility rate (TFR)
            tfr = np.clip(7.5 - 0.8 * (log_gdppc - 5.5) - 0.08 * elapsed + rng.normal(0, 0.3), 1.5, 8)

            # Access to electricity (%)
            electricity = np.clip(
                25 + 10 * (log_gdppc - 5.5) + 1.5 * elapsed + rng.normal(0, 3), 5, 100
            )

            # Access to improved sanitation (%)
            sanitation = np.clip(
                20 + 8 * (log_gdppc - 5.5) + 1.0 * elapsed + rng.normal(0, 3), 5, 100
            )

            # Agriculture % of GDP (declines with development)
            agri_gdp = np.clip(
                50 - 5 * (log_gdppc - 5.5) - 0.3 * elapsed + rng.normal(0, 2), 3, 60
            )

            # Trade openness (exports + imports % GDP)
            trade = np.clip(40 + rng.normal(0, 12) + 0.3 * elapsed, 15, 120)

            # Poverty headcount ($2.15/day)
            poverty = np.clip(
                75 - 10 * (log_gdppc - 5.5) - 1.0 * elapsed + rng.normal(0, 3), 0, 90
            )

            # Gini coefficient
            gini = np.clip(38 + rng.normal(0, 5) + 2 * (region == "Latin America & Caribbean"), 25, 65)

            # Government expenditure on education (% GDP)
            educ_spending = np.clip(3.5 + rng.normal(0, 1.0), 1, 8)

            # Mobile subscriptions per 100 people
            mobile = np.clip(
                5 * (1.25 ** min(elapsed, 15)) + rng.normal(0, 5), 0, 150
            )

            rows.append({
                "country": country,
                "iso3": meta["iso"],
                "region": region,
                "year": year,
                "gdp_per_capita_usd": round(gdppc, 1),
                "log_gdp_per_capita": round(log_gdppc, 4),
                "life_expectancy": round(le, 1),
                "infant_mortality_per_1000": round(imr, 1),
                "under5_mortality_per_1000": round(u5mr, 1),
                "primary_enrollment_pct": round(primary_enroll, 1),
                "secondary_enrollment_pct": round(secondary_enroll, 1),
                "adult_literacy_pct": round(literacy, 1),
                "fertility_rate": round(tfr, 2),
                "electricity_access_pct": round(electricity, 1),
                "sanitation_access_pct": round(sanitation, 1),
                "agriculture_pct_gdp": round(agri_gdp, 1),
                "trade_openness_pct_gdp": round(trade, 1),
                "poverty_headcount_215": round(poverty, 1),
                "gini_coefficient": round(gini, 1),
                "education_spending_pct_gdp": round(educ_spending, 1),
                "mobile_subscriptions_per_100": round(mobile, 1),
            })

    df = pd.DataFrame(rows)

    # Inject WDI-like missingness (certain indicators missing for certain years)
    df = _inject_wdi_gaps(df, rng)

    return df


def _inject_wdi_gaps(df, rng):
    """Mimic WDI's pattern: poverty & Gini sparse, GDP almost complete."""
    sparse_cols = ["poverty_headcount_215", "gini_coefficient", "education_spending_pct_gdp"]
    medium_cols = ["adult_literacy_pct", "sanitation_access_pct"]
    dense_cols = ["secondary_enrollment_pct", "fertility_rate"]

    from .utils import inject_missing
    df = inject_missing(df, sparse_cols, [0.30, 0.35, 0.15], rng=rng, mechanism="MCAR")
    df = inject_missing(df, medium_cols, [0.08, 0.06], rng=rng, mechanism="MCAR")
    df = inject_missing(df, dense_cols, [0.03, 0.02], rng=rng, mechanism="MCAR")
    return df
