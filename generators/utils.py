"""
Shared utilities for realistic data generation.
Provides correlated draws, missing-data injection, geographic scaffolding,
and reproducible seeding used by all dataset generators.
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Geography helpers
# ---------------------------------------------------------------------------

COUNTRIES = {
    "Kenya": {"iso": "KEN", "region": "Sub-Saharan Africa", "pop_m": 54, "gdppc": 1840},
    "Tanzania": {"iso": "TZA", "region": "Sub-Saharan Africa", "pop_m": 63, "gdppc": 1100},
    "Uganda": {"iso": "UGA", "region": "Sub-Saharan Africa", "pop_m": 47, "gdppc": 860},
    "Ethiopia": {"iso": "ETH", "region": "Sub-Saharan Africa", "pop_m": 120, "gdppc": 940},
    "Rwanda": {"iso": "RWA", "region": "Sub-Saharan Africa", "pop_m": 13, "gdppc": 830},
    "Nigeria": {"iso": "NGA", "region": "Sub-Saharan Africa", "pop_m": 220, "gdppc": 2060},
    "Ghana": {"iso": "GHA", "region": "Sub-Saharan Africa", "pop_m": 33, "gdppc": 2360},
    "Senegal": {"iso": "SEN", "region": "Sub-Saharan Africa", "pop_m": 17, "gdppc": 1600},
    "Mozambique": {"iso": "MOZ", "region": "Sub-Saharan Africa", "pop_m": 33, "gdppc": 500},
    "Malawi": {"iso": "MWI", "region": "Sub-Saharan Africa", "pop_m": 20, "gdppc": 630},
    "India": {"iso": "IND", "region": "South Asia", "pop_m": 1420, "gdppc": 2380},
    "Bangladesh": {"iso": "BGD", "region": "South Asia", "pop_m": 170, "gdppc": 2690},
    "Nepal": {"iso": "NPL", "region": "South Asia", "pop_m": 30, "gdppc": 1340},
    "Pakistan": {"iso": "PAK", "region": "South Asia", "pop_m": 230, "gdppc": 1500},
    "Sri Lanka": {"iso": "LKA", "region": "South Asia", "pop_m": 22, "gdppc": 3830},
    "Cambodia": {"iso": "KHM", "region": "East Asia & Pacific", "pop_m": 17, "gdppc": 1690},
    "Vietnam": {"iso": "VNM", "region": "East Asia & Pacific", "pop_m": 99, "gdppc": 4120},
    "Philippines": {"iso": "PHL", "region": "East Asia & Pacific", "pop_m": 114, "gdppc": 3460},
    "Indonesia": {"iso": "IDN", "region": "East Asia & Pacific", "pop_m": 276, "gdppc": 4290},
    "Guatemala": {"iso": "GTM", "region": "Latin America & Caribbean", "pop_m": 18, "gdppc": 5020},
    "Honduras": {"iso": "HND", "region": "Latin America & Caribbean", "pop_m": 10, "gdppc": 2770},
    "Bolivia": {"iso": "BOL", "region": "Latin America & Caribbean", "pop_m": 12, "gdppc": 3500},
    "Peru": {"iso": "PER", "region": "Latin America & Caribbean", "pop_m": 34, "gdppc": 6680},
    "Colombia": {"iso": "COL", "region": "Latin America & Caribbean", "pop_m": 52, "gdppc": 6100},
    "Morocco": {"iso": "MAR", "region": "Middle East & North Africa", "pop_m": 37, "gdppc": 3800},
}

DISTRICTS = [
    "Kilifi", "Mombasa", "Nairobi", "Kisumu", "Nakuru", "Machakos",
    "Dodoma", "Mwanza", "Arusha", "Dar es Salaam", "Kampala", "Gulu",
    "Addis Ababa", "Hawassa", "Bahir Dar", "Kigali", "Musanze",
    "Lagos", "Kano", "Abuja", "Accra", "Kumasi", "Tamale",
    "Dakar", "Thies", "Maputo", "Nampula", "Lilongwe", "Blantyre",
    "Mumbai", "Delhi", "Kolkata", "Chennai", "Dhaka", "Chittagong",
    "Kathmandu", "Pokhara", "Lahore", "Karachi", "Colombo",
    "Phnom Penh", "Siem Reap", "Hanoi", "Ho Chi Minh", "Manila",
    "Jakarta", "Surabaya", "Guatemala City", "Tegucigalpa",
    "La Paz", "Cochabamba", "Lima", "Cusco", "Bogota", "Medellin",
    "Casablanca", "Marrakech",
]


def pick_districts(rng: np.random.Generator, n: int, urban_share: float = 0.35):
    """Assign districts with an urban/rural flag."""
    districts = rng.choice(DISTRICTS, size=n)
    urban = rng.random(n) < urban_share
    return districts, urban


# ---------------------------------------------------------------------------
# Correlated draws
# ---------------------------------------------------------------------------

def correlated_normal(rng: np.random.Generator, n: int, means: list,
                      stds: list, corr_matrix: np.ndarray) -> np.ndarray:
    """Draw n samples from a multivariate normal with given correlations."""
    k = len(means)
    D = np.diag(stds)
    cov = D @ corr_matrix @ D
    return rng.multivariate_normal(means, cov, size=n)


# ---------------------------------------------------------------------------
# Missing data injection
# ---------------------------------------------------------------------------

def inject_missing(df: pd.DataFrame, columns: list,
                   rates: Optional[list] = None,
                   rng: Optional[np.random.Generator] = None,
                   mechanism: str = "MCAR") -> pd.DataFrame:
    """
    Inject realistic missing values.
    mechanism: MCAR (random), MAR (correlated with another column), or
               MNAR (correlated with own value).
    """
    rng = rng or np.random.default_rng(42)
    if rates is None:
        rates = [0.05] * len(columns)

    df = df.copy()
    for col, rate in zip(columns, rates):
        if mechanism == "MCAR":
            mask = rng.random(len(df)) < rate
        elif mechanism == "MAR":
            # higher missingness for lower-value rows of first numeric col
            ref = df.select_dtypes(include="number").iloc[:, 0]
            prob = rate * 2 * (1 - (ref - ref.min()) / (ref.max() - ref.min() + 1e-9))
            mask = rng.random(len(df)) < prob.values
        elif mechanism == "MNAR":
            vals = pd.to_numeric(df[col], errors="coerce")
            if vals.notna().any():
                prob = rate * 2 * (1 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-9))
                mask = rng.random(len(df)) < prob.fillna(rate).values
            else:
                mask = rng.random(len(df)) < rate
        else:
            mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# Realistic ID generators
# ---------------------------------------------------------------------------

def household_ids(rng: np.random.Generator, n: int, prefix: str = "HH") -> list:
    """Generate unique household IDs like HH-001234."""
    return [f"{prefix}-{i:06d}" for i in range(1, n + 1)]


def individual_ids(rng: np.random.Generator, n: int, prefix: str = "IND") -> list:
    return [f"{prefix}-{i:07d}" for i in range(1, n + 1)]


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def random_dates(rng: np.random.Generator, n: int,
                 start: str = "2018-01-01", end: str = "2024-12-31") -> pd.Series:
    """Generate random dates in a range."""
    start_ts = pd.Timestamp(start).value // 10**9
    end_ts = pd.Timestamp(end).value // 10**9
    timestamps = rng.integers(start_ts, end_ts, size=n)
    return pd.to_datetime(timestamps, unit="s").normalize()


# ---------------------------------------------------------------------------
# PPP conversion helpers
# ---------------------------------------------------------------------------

PPP_FACTORS = {
    "KEN": 50.4, "TZA": 891.3, "UGA": 1239.0, "ETH": 17.7, "RWA": 371.0,
    "NGA": 167.5, "GHA": 3.44, "SEN": 237.0, "MOZ": 25.3, "MWI": 312.0,
    "IND": 22.9, "BGD": 34.8, "NPL": 41.2, "PAK": 49.1, "LKA": 71.5,
    "KHM": 1595.0, "VNM": 8160.0, "PHL": 19.2, "IDN": 5330.0,
    "GTM": 4.07, "HND": 12.3, "BOL": 3.46, "PER": 1.73, "COL": 1543.0,
    "MAR": 3.83,
}
