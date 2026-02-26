"""
Generator: ODA & Aid Effectiveness
────────────────────────────────────
Simulates aid-flow-level data on Official Development Assistance (ODA) and
Paris Declaration effectiveness indicators, covering donor-recipient
relationships, disbursement types, sector allocation, Paris principles
(ownership, alignment, harmonisation, results, mutual accountability),
fragmentation metrics, conditionality, coordination mechanisms, and
sustainability measures.

Rows: one per aid flow record (~12k), spanning years 2010-2025.

Realistic features:
  • Aid flows more to poorer countries but also to strategic/geopolitical
    interests (population-weighted)
  • Disbursement types: grants, concessional loans, technical cooperation,
    budget support, humanitarian
  • Paris Declaration indicators: country ownership, alignment with national
    plans, use of country systems, mutual accountability
  • Fragmentation: donor concentration (HHI), tied aid, project size
  • Conditionality: policy, fiduciary, results-based
  • Coordination: joint programmes, pooled funding, lead donor arrangements
  • Sustainability: exit strategies, local capacity, technology transfer
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_flows: int = 12000, seed: int = 811) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_flows

    # --- IDs ---
    flow_ids = [f"AID-{i:06d}" for i in range(1, n + 1)]

    # --- Donor countries (DAC members + emerging donors) ---
    donor_countries_list = [
        "United States", "United Kingdom", "Germany", "France", "Japan",
        "Canada", "Netherlands", "Sweden", "Norway", "Denmark",
        "Australia", "Switzerland", "South Korea", "Italy", "Spain",
        "Belgium", "Finland", "Ireland", "Austria", "New Zealand",
        "China", "India", "Saudi Arabia", "UAE", "Turkey",
    ]
    donor_probs = np.array([
        0.15, 0.08, 0.10, 0.07, 0.08,
        0.04, 0.04, 0.04, 0.03, 0.03,
        0.03, 0.03, 0.03, 0.03, 0.02,
        0.02, 0.02, 0.02, 0.01, 0.01,
        0.04, 0.02, 0.02, 0.02, 0.02,
    ])
    donor_probs = donor_probs / donor_probs.sum()
    donor_country = rng.choice(donor_countries_list, n, p=donor_probs)

    # --- Recipient countries (from COUNTRIES, weighted by pop & inverse GDP) ---
    recipient_list = list(COUNTRIES.keys())
    # More aid to larger, poorer countries (but with noise for geopolitics)
    recipient_weights = np.array([
        COUNTRIES[c]["pop_m"] / (COUNTRIES[c]["gdppc"] ** 0.5)
        for c in recipient_list
    ])
    recipient_weights = recipient_weights / recipient_weights.sum()
    recipient_country = rng.choice(recipient_list, n, p=recipient_weights)

    # --- Year (2010-2025) ---
    year = rng.integers(2010, 2026, n)

    # --- Aid flow characteristics ---
    disbursement_type = rng.choice(
        ["grant", "concessional_loan", "technical_cooperation",
         "budget_support", "humanitarian"],
        n, p=[0.35, 0.15, 0.18, 0.12, 0.20],
    )
    channel = rng.choice(
        ["bilateral", "multilateral_un", "multilateral_mdb", "ngo", "private"],
        n, p=[0.35, 0.20, 0.15, 0.20, 0.10],
    )

    # ODA amount: log-normal, larger for bilateral, budget support
    is_bilateral = (channel == "bilateral").astype(float)
    is_budget_support = (disbursement_type == "budget_support").astype(float)
    is_humanitarian = (disbursement_type == "humanitarian").astype(float)

    oda_amount_usd = np.round(
        np.exp(
            13.0 + 0.5 * is_bilateral + 0.8 * is_budget_support
            + 0.3 * is_humanitarian + rng.normal(0, 1.2, n)
        ),
        -2,
    )

    # --- Sector ---
    sector = rng.choice(
        ["health", "education", "governance", "agriculture", "infrastructure",
         "environment", "humanitarian", "social_protection", "trade", "multisector"],
        n, p=[0.15, 0.13, 0.10, 0.10, 0.12, 0.08, 0.12, 0.08, 0.05, 0.07],
    )

    # Latent quality of aid relationship (drives Paris indicators)
    relationship_quality = (
        rng.normal(0, 1, n)
        + 0.2 * is_budget_support
        - 0.15 * is_humanitarian
        + 0.1 * (year - 2010) / 15  # slight improvement over time
    )

    # --- Paris Declaration principles ---
    country_ownership_score = np.clip(
        rng.normal(
            2.8 + 0.4 * relationship_quality + 0.3 * is_budget_support, 0.8, n,
        ),
        1, 5,
    ).round(1)

    alignment_with_national_plan = rng.binomial(
        1, _logistic(0.55, 0.3 * relationship_quality + 0.2 * is_budget_support, 0.4)
    )
    uses_country_systems = rng.binomial(
        1, _logistic(0.35, 0.3 * relationship_quality + 0.35 * is_budget_support, 0.5)
    )
    mutual_accountability = rng.binomial(
        1, _logistic(0.45, 0.25 * relationship_quality + 0.15 * is_budget_support, 0.4)
    )
    results_orientation_score = np.clip(
        rng.normal(
            3.0 + 0.35 * relationship_quality + 0.1 * (year - 2010) / 15, 0.8, n,
        ),
        1, 5,
    ).round(1)

    # --- Fragmentation indicators ---
    n_donors_in_sector = np.clip(
        rng.poisson(np.clip(6 + 3 * rng.normal(0, 1, n), 1, 30), n),
        1, 35,
    )
    # Herfindahl-Hirschman Index for donor concentration (0-1)
    donor_concentration_index = np.clip(
        rng.beta(2, 5, n) + 0.05 * (1 / n_donors_in_sector),
        0.01, 1.0,
    ).round(3)

    project_size_usd = oda_amount_usd.copy()

    is_tied_aid = rng.binomial(
        1, _logistic(0.15, -0.1 * relationship_quality - 0.1 * (year - 2010) / 15, 0.3)
    )

    # --- Effectiveness ---
    # Disbursement ratio: committed vs. actually disbursed
    disbursement_ratio = np.clip(
        rng.normal(
            0.78 + 0.08 * relationship_quality + 0.05 * is_budget_support, 0.15, n,
        ),
        0.20, 1.0,
    ).round(2)

    on_budget = rng.binomial(
        1, _logistic(0.40, 0.25 * relationship_quality + 0.3 * is_budget_support, 0.4)
    )
    predictability_score = np.clip(
        rng.normal(
            3.0 + 0.3 * relationship_quality + 0.2 * is_budget_support
            - 0.15 * is_humanitarian, 0.9, n,
        ),
        1, 5,
    ).round(1)

    technical_assistance_pct = np.clip(
        rng.normal(
            20 + 15 * (disbursement_type == "technical_cooperation").astype(float)
            - 5 * is_budget_support, 12, n,
        ),
        0, 80,
    ).round(1)

    # --- Conditionality ---
    has_conditionality = rng.binomial(
        1, _logistic(0.40, 0.1 * is_bilateral + 0.15 * is_budget_support, 0.3)
    )
    conditionality_type = np.where(
        has_conditionality,
        rng.choice(
            ["policy", "fiduciary", "results"],
            n, p=[0.40, 0.35, 0.25],
        ),
        "none",
    )
    conditionality_met = np.where(
        has_conditionality,
        rng.binomial(1, _logistic(0.60, 0.2 * relationship_quality, 0.3)),
        0,
    )

    # --- Coordination ---
    joint_programme = rng.binomial(
        1, _logistic(0.20, 0.15 * relationship_quality + 0.1 * (year - 2010) / 15, 0.3)
    )
    pooled_funding = rng.binomial(
        1, _logistic(0.15, 0.12 * relationship_quality + 0.1 * is_budget_support, 0.3)
    )
    lead_donor_arrangement = rng.binomial(
        1, _logistic(0.18, 0.1 * relationship_quality + 0.05 * joint_programme, 0.3)
    )

    # --- Sustainability ---
    exit_strategy = rng.binomial(
        1, _logistic(0.35, 0.2 * relationship_quality + 0.1 * (year - 2010) / 15, 0.3)
    )
    local_capacity_built = rng.binomial(
        1, _logistic(
            0.40, 0.2 * relationship_quality + 0.15 * uses_country_systems
            + 0.1 * country_ownership_score / 5, 0.4,
        )
    )
    technology_transferred = rng.binomial(
        1, _logistic(
            0.20, 0.1 * relationship_quality
            + 0.15 * (disbursement_type == "technical_cooperation").astype(float), 0.3,
        )
    )

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "flow_id": flow_ids,
        "donor_country": donor_country,
        "recipient_country": recipient_country,
        "year": year,
        # Aid flow
        "oda_amount_usd": oda_amount_usd,
        "disbursement_type": disbursement_type,
        "channel": channel,
        # Sector
        "sector": sector,
        # Paris principles
        "country_ownership_score": country_ownership_score,
        "alignment_with_national_plan": alignment_with_national_plan,
        "uses_country_systems": uses_country_systems,
        "mutual_accountability": mutual_accountability,
        "results_orientation_score": results_orientation_score,
        # Fragmentation
        "n_donors_in_sector": n_donors_in_sector,
        "donor_concentration_index": donor_concentration_index,
        "project_size_usd": project_size_usd,
        "is_tied_aid": is_tied_aid,
        # Effectiveness
        "disbursement_ratio": disbursement_ratio,
        "on_budget": on_budget,
        "predictability_score": predictability_score,
        "technical_assistance_pct": technical_assistance_pct,
        # Conditionality
        "has_conditionality": has_conditionality,
        "conditionality_type": conditionality_type,
        "conditionality_met": conditionality_met,
        # Coordination
        "joint_programme": joint_programme,
        "pooled_funding": pooled_funding,
        "lead_donor_arrangement": lead_donor_arrangement,
        # Sustainability
        "exit_strategy": exit_strategy,
        "local_capacity_built": local_capacity_built,
        "technology_transferred": technology_transferred,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "oda_amount_usd", "country_ownership_score", "results_orientation_score",
            "donor_concentration_index", "disbursement_ratio",
            "predictability_score", "technical_assistance_pct",
        ],
        rates=[0.05, 0.07, 0.06, 0.08, 0.06, 0.07, 0.05],
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
