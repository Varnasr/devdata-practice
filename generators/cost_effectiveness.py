"""
Generator: Programme Costing & CEA
───────────────────────────────────
Simulates programme-level cost data for cost-effectiveness analysis across
health, education, nutrition, WASH, livelihoods, and social protection sectors.

Rows: ~15k programme observations.

Realistic features:
  • Lognormal cost distributions
  • Personnel 40-65%, overhead 8-25%
  • Cost breakdowns (personnel, materials, transport, overhead, monitoring)
  • ICER, DALY, QALY metrics for health programmes
  • Pilot vs at-scale with economies of scale
  • Multiple implementer types
  • MCAR missingness mechanism
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_programmes: int = 15000, seed: int = 802) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_programmes

    # --- Programme identifiers ---
    ids = [f"PRG-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.42)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    sectors = ["health", "education", "nutrition", "wash", "livelihoods", "social_protection"]
    sector = rng.choice(sectors, n, p=[0.22, 0.20, 0.12, 0.16, 0.15, 0.15])

    programme_types_map = {
        "health": ["vaccination", "maternal_health", "community_health_worker",
                    "hiv_prevention", "malaria_control", "mental_health"],
        "education": ["school_construction", "teacher_training", "school_feeding",
                      "learning_materials", "girls_scholarships", "adult_literacy"],
        "nutrition": ["supplementary_feeding", "micronutrient_supplement",
                      "growth_monitoring", "nutrition_education", "fortification"],
        "wash": ["borehole_drilling", "latrine_construction", "hygiene_promotion",
                 "water_treatment", "piped_water_extension"],
        "livelihoods": ["cash_transfer", "skills_training", "microfinance",
                        "agricultural_inputs", "market_linkages"],
        "social_protection": ["unconditional_cash", "conditional_cash", "public_works",
                              "pension", "disability_grant", "child_grant"],
    }
    programme_type = np.array([
        rng.choice(programme_types_map[s]) for s in sector
    ])

    implementation_year = rng.integers(2015, 2025, n)
    duration_months = np.clip(rng.normal(18, 8, n), 3, 60).astype(int)

    # Implementer
    implementer_types = ["government", "ingo", "local_ngo", "un_agency"]
    implementer_type = rng.choice(implementer_types, n, p=[0.30, 0.28, 0.22, 0.20])

    # Scale
    is_pilot = rng.binomial(1, 0.30, n)
    economies_of_scale = np.where(is_pilot, 1.0,
        np.clip(rng.normal(0.75, 0.10, n), 0.50, 0.95)
    ).round(2)

    # --- Beneficiaries ---
    target_beneficiaries = np.clip(
        rng.lognormal(7.5 - 1.5 * is_pilot, 1.2, n), 50, 500000
    ).astype(int)
    coverage_rate = np.clip(rng.beta(5, 2, n), 0.30, 1.0).round(2)
    actual_beneficiaries = (target_beneficiaries * coverage_rate).astype(int)

    # --- Costs (lognormal, USD) ---
    base_log_cost = 10.5 + 0.8 * (1 - is_pilot) + rng.normal(0, 0.5, n)
    total_cost_usd = np.round(np.exp(base_log_cost) * economies_of_scale, 2)

    # Cost component ratios (must sum to 1)
    personnel_ratio = np.clip(rng.normal(0.52, 0.06, n), 0.40, 0.65).round(3)
    overhead_ratio = np.clip(rng.normal(0.15, 0.04, n), 0.08, 0.25).round(3)
    monitoring_share = np.clip(rng.normal(0.08, 0.02, n), 0.03, 0.15).round(3)
    remaining = np.clip(1.0 - personnel_ratio - overhead_ratio - monitoring_share, 0.05, 1.0)
    materials_share = np.clip(rng.beta(3, 3, n) * remaining, 0.01, remaining)
    transport_share = remaining - materials_share

    personnel_cost_usd = np.round(total_cost_usd * personnel_ratio, 2)
    overhead_cost_usd = np.round(total_cost_usd * overhead_ratio, 2)
    monitoring_cost_usd = np.round(total_cost_usd * monitoring_share, 2)
    materials_cost_usd = np.round(total_cost_usd * materials_share, 2)
    transport_cost_usd = np.round(total_cost_usd * transport_share, 2)

    # Unit costs
    cost_per_beneficiary_usd = np.round(
        total_cost_usd / np.maximum(actual_beneficiaries, 1), 2
    )

    # --- Outcomes ---
    baseline_value = np.clip(rng.normal(0.35, 0.15, n), 0.0, 0.90).round(3)
    effect_size = np.clip(
        rng.normal(0.12 + 0.05 * (1 - is_pilot), 0.08, n), -0.05, 0.50
    ).round(3)
    endline_value = np.clip(baseline_value + effect_size, 0.0, 1.0).round(3)
    primary_outcome_value = endline_value

    cost_per_outcome_usd = np.where(
        effect_size > 0,
        np.round(total_cost_usd / np.maximum(actual_beneficiaries * effect_size, 1), 2),
        np.nan,
    )

    # --- CEA metrics (health-sector specific, NaN for others) ---
    is_health = (sector == "health").astype(float)
    daly_averted = np.where(
        sector == "health",
        np.clip(rng.lognormal(1.5 + 0.3 * effect_size * 10, 0.8, n), 0.1, 5000),
        np.nan,
    )
    daly_averted = np.where(sector == "health", np.round(daly_averted, 1), np.nan)

    qaly_gained = np.where(
        sector == "health",
        np.clip(rng.lognormal(0.8 + 0.2 * effect_size * 10, 0.7, n), 0.05, 3000),
        np.nan,
    )
    qaly_gained = np.where(sector == "health", np.round(qaly_gained, 1), np.nan)

    icer = np.where(
        sector == "health",
        np.round(total_cost_usd / np.maximum(np.where(np.isnan(daly_averted), 1, daly_averted), 0.1), 2),
        np.nan,
    )

    df = pd.DataFrame({
        "programme_id": ids,
        "country": countries,
        "district": districts,
        "sector": sector,
        "programme_type": programme_type,
        "implementer_type": implementer_type,
        "implementation_year": implementation_year,
        "duration_months": duration_months,
        "is_pilot": is_pilot,
        "economies_of_scale": economies_of_scale,
        "target_beneficiaries": target_beneficiaries,
        "actual_beneficiaries": actual_beneficiaries,
        "coverage_rate": coverage_rate,
        "total_cost_usd": total_cost_usd,
        "personnel_cost_usd": personnel_cost_usd,
        "materials_cost_usd": materials_cost_usd,
        "transport_cost_usd": transport_cost_usd,
        "overhead_cost_usd": overhead_cost_usd,
        "monitoring_cost_usd": monitoring_cost_usd,
        "personnel_ratio": personnel_ratio,
        "overhead_ratio": overhead_ratio,
        "cost_per_beneficiary_usd": cost_per_beneficiary_usd,
        "cost_per_outcome_usd": cost_per_outcome_usd,
        "baseline_value": baseline_value,
        "endline_value": endline_value,
        "primary_outcome_value": primary_outcome_value,
        "effect_size": effect_size,
        "icer": icer,
        "daly_averted": daly_averted,
        "qaly_gained": qaly_gained,
    })

    df = inject_missing(df,
        columns=["total_cost_usd", "effect_size", "cost_per_beneficiary_usd",
                 "coverage_rate", "baseline_value"],
        rates=[0.03, 0.08, 0.04, 0.05, 0.06],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1 - 1e-6) / (1 - np.clip(base, 1e-6, 1 - 1e-6))) + slope * np.asarray(z))
