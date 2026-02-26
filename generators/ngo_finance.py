"""
Generator: NGO Programme Finance
─────────────────────────────────
Simulates programme-level financial and management data for non-governmental
organisations, covering organisation characteristics, funding sources,
programme delivery, budget breakdowns, financial health indicators,
compliance and audit status, effectiveness measures, and partnership
structures.

Rows: one per programme record (~10k), with cross-sectional data.

Realistic features:
  • Organisation types: local NGO, INGO, CBO, faith-based, social enterprise
  • Budget breakdowns with realistic overhead ratios (admin 10-25%)
  • Financial health: burn rate, reserves, cost recovery, sustainability
  • Compliance indicators: audit status, donor reporting timeliness
  • Effectiveness: outcome targets, beneficiary satisfaction, value for money
  • Overhead debate: admin costs realistically 10-25%, with pressure to
    report lower (some orgs under-report)
  • Reached population constrained by target population
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_programmes: int = 10000, seed: int = 810) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_programmes

    # --- IDs & geography ---
    programme_ids = [f"PRG-{i:06d}" for i in range(1, n + 1)]
    org_ids = rng.choice([f"ORG-{i:04d}" for i in range(1, 801)], n)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # --- Organisation characteristics ---
    org_type = rng.choice(
        ["local_ngo", "ingo", "cbo", "faith_based", "social_enterprise"],
        n, p=[0.30, 0.25, 0.20, 0.15, 0.10],
    )
    is_ingo = (org_type == "ingo").astype(float)
    is_cbo = (org_type == "cbo").astype(float)
    is_social_ent = (org_type == "social_enterprise").astype(float)

    org_size = rng.choice(
        ["small", "medium", "large"],
        n, p=[0.40, 0.35, 0.25],
    )
    size_factor = np.where(org_size == "large", 1.0,
                  np.where(org_size == "medium", 0.5, 0.0))

    org_age_years = np.clip(
        rng.exponential(8 + 5 * is_ingo + 3 * size_factor, n), 1, 60
    ).astype(int)

    n_staff = np.clip(
        rng.poisson(np.clip(15 + 40 * size_factor + 20 * is_ingo, 1, 500), n),
        1, 500,
    )
    n_volunteers = np.clip(
        rng.poisson(np.clip(10 + 5 * size_factor + 15 * is_cbo, 1, 200), n),
        0, 300,
    )

    # Latent organisational capacity (drives many outcomes)
    capacity_latent = (
        rng.normal(0, 1, n)
        + 0.3 * size_factor
        + 0.2 * is_ingo
        + 0.1 * org_age_years / 20
    )

    annual_budget_usd = np.round(
        np.exp(10.5 + 1.2 * size_factor + 0.6 * is_ingo
               + 0.3 * capacity_latent + rng.normal(0, 0.5, n)),
        -2,
    )

    # --- Funding ---
    n_donors = np.clip(
        rng.poisson(np.clip(3 + 4 * size_factor + 2 * is_ingo, 0.5, 30), n),
        1, 30,
    )
    primary_donor_type = rng.choice(
        ["bilateral", "multilateral", "foundation", "corporate",
         "individual", "government"],
        n, p=[0.25, 0.15, 0.22, 0.12, 0.14, 0.12],
    )
    funding_secured_pct = np.clip(
        rng.normal(72 + 8 * capacity_latent + 5 * is_ingo, 18, n),
        10, 100,
    ).round(1)
    funding_gap_usd = np.round(
        annual_budget_usd * (100 - funding_secured_pct) / 100, -1
    )

    # --- Programme details ---
    sector = rng.choice(
        ["health", "education", "wash", "livelihoods", "protection",
         "governance", "nutrition", "drr"],
        n, p=[0.18, 0.16, 0.12, 0.15, 0.10, 0.10, 0.10, 0.09],
    )
    programme_duration_months = rng.choice(
        [6, 12, 18, 24, 36, 48, 60],
        n, p=[0.08, 0.18, 0.20, 0.22, 0.18, 0.10, 0.04],
    )
    target_population = np.clip(
        rng.lognormal(8.5 + 0.5 * size_factor, 1.2, n).astype(int),
        50, 500000,
    )
    reach_rate = np.clip(
        rng.normal(0.70 + 0.08 * capacity_latent + 0.05 * programme_duration_months / 60, 0.18, n),
        0.10, 1.0,
    )
    reached_population = (target_population * reach_rate).astype(int)

    # --- Budget breakdown (percentages should sum close to 100) ---
    # Realistic overhead debate: admin costs 10-25% with pressure to report lower
    admin_pct_raw = np.clip(
        rng.normal(17 + 3 * is_ingo - 2 * is_cbo, 5, n), 5, 35
    )
    # Some orgs under-report admin (pressure to show low overhead)
    under_reports = rng.binomial(1, 0.25, n)
    admin_pct = np.where(
        under_reports,
        np.clip(admin_pct_raw * rng.uniform(0.5, 0.8, n), 3, 25),
        admin_pct_raw,
    ).round(1)

    monitoring_pct = np.clip(
        rng.normal(6 + 1.5 * capacity_latent, 2.5, n), 1, 15
    ).round(1)
    indirect_cost_pct = np.clip(
        rng.normal(8 + 2 * is_ingo, 3, n), 2, 18
    ).round(1)
    personnel_pct = np.clip(
        rng.normal(35 + 3 * size_factor, 8, n), 15, 60
    ).round(1)
    # Programme activities gets the remainder
    programme_activities_pct = np.clip(
        100 - personnel_pct - admin_pct - monitoring_pct - indirect_cost_pct,
        5, 55,
    ).round(1)

    # --- Financial health ---
    burn_rate_pct = np.clip(
        rng.normal(85 + 5 * (1 - capacity_latent * 0.2), 12, n),
        40, 120,
    ).round(1)
    months_of_reserves = np.clip(
        rng.exponential(np.clip(3 + 2 * capacity_latent + 1.5 * is_ingo, 0.1, 30), n),
        0, 24,
    ).round(1)
    cost_recovery_ratio = np.clip(
        rng.normal(0.85 + 0.08 * capacity_latent + 0.1 * is_social_ent, 0.15, n),
        0.20, 1.50,
    ).round(2)
    sustainability_score = np.clip(
        rng.normal(
            2.8 + 0.4 * capacity_latent + 0.3 * is_social_ent
            + 0.1 * org_age_years / 20, 0.8, n,
        ),
        1, 5,
    ).round(1)

    # --- Compliance ---
    audit_completed = rng.binomial(
        1, _logistic(0.75, 0.3 * capacity_latent + 0.2 * is_ingo, 0.5)
    )
    audit_qualified = np.where(
        audit_completed,
        rng.binomial(1, _logistic(0.12, -0.2 * capacity_latent, 0.3)),
        0,
    )
    donor_reporting_on_time = rng.binomial(
        1, _logistic(0.70, 0.25 * capacity_latent + 0.15 * is_ingo, 0.4)
    )
    financial_transparency_score = np.clip(
        rng.normal(
            3.0 + 0.4 * capacity_latent + 0.3 * audit_completed
            + 0.2 * is_ingo, 0.8, n,
        ),
        1, 5,
    ).round(1)

    # --- Effectiveness ---
    outcome_target_met = rng.binomial(
        1, _logistic(0.55, 0.2 * capacity_latent + 0.1 * monitoring_pct / 10, 0.4)
    )
    beneficiary_satisfaction = np.clip(
        rng.normal(
            3.3 + 0.3 * capacity_latent + 0.15 * reach_rate
            + 0.1 * programme_duration_months / 60, 0.7, n,
        ),
        1, 5,
    ).round(1)
    vfm_score = np.clip(
        rng.normal(
            3.0 + 0.25 * capacity_latent + 0.1 * outcome_target_met
            - 0.1 * admin_pct / 20, 0.8, n,
        ),
        1, 5,
    ).round(1)

    # --- Partnerships ---
    n_partners = np.clip(
        rng.poisson(np.clip(2 + 3 * size_factor + 1.5 * is_ingo, 0.5, 20), n),
        0, 25,
    )
    local_partner_led = rng.binomial(
        1, _logistic(0.35, 0.2 * is_cbo - 0.15 * is_ingo + 0.1 * capacity_latent, 0.4)
    )
    community_contribution_pct = np.clip(
        rng.normal(8 + 5 * is_cbo - 2 * is_ingo + 2 * capacity_latent, 5, n),
        0, 40,
    ).round(1)

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "programme_id": programme_ids,
        "org_id": org_ids,
        "country": countries,
        "org_type": org_type,
        # Organisation
        "org_size": org_size,
        "org_age_years": org_age_years,
        "n_staff": n_staff,
        "n_volunteers": n_volunteers,
        "annual_budget_usd": annual_budget_usd,
        # Funding
        "n_donors": n_donors,
        "primary_donor_type": primary_donor_type,
        "funding_secured_pct": funding_secured_pct,
        "funding_gap_usd": funding_gap_usd,
        # Programme
        "sector": sector,
        "programme_duration_months": programme_duration_months,
        "target_population": target_population,
        "reached_population": reached_population,
        # Budget breakdown
        "personnel_pct": personnel_pct,
        "programme_activities_pct": programme_activities_pct,
        "admin_pct": admin_pct,
        "monitoring_pct": monitoring_pct,
        "indirect_cost_pct": indirect_cost_pct,
        # Financial health
        "burn_rate_pct": burn_rate_pct,
        "months_of_reserves": months_of_reserves,
        "cost_recovery_ratio": cost_recovery_ratio,
        "sustainability_score": sustainability_score,
        # Compliance
        "audit_completed": audit_completed,
        "audit_qualified": audit_qualified,
        "donor_reporting_on_time": donor_reporting_on_time,
        "financial_transparency_score": financial_transparency_score,
        # Effectiveness
        "outcome_target_met": outcome_target_met,
        "beneficiary_satisfaction": beneficiary_satisfaction,
        "vfm_score": vfm_score,
        # Partnerships
        "n_partners": n_partners,
        "local_partner_led": local_partner_led,
        "community_contribution_pct": community_contribution_pct,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "annual_budget_usd", "funding_secured_pct", "funding_gap_usd",
            "reached_population", "burn_rate_pct", "months_of_reserves",
            "cost_recovery_ratio", "beneficiary_satisfaction", "vfm_score",
            "community_contribution_pct",
        ],
        rates=[0.06, 0.05, 0.05, 0.04, 0.07, 0.08,
               0.06, 0.09, 0.08, 0.07],
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
