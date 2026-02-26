"""
Generator: Social Capital & Collective Action
──────────────────────────────────────────────
Simulates individual-level data on community development, social capital
measurement, participatory governance, and community-driven development
programmes.

Rows: ~18k individuals.

Realistic features:
  • Bonding and bridging social capital measurement
  • Group memberships and types
  • Trust indicators: neighbors, strangers, local leaders
  • Collective action participation and types
  • Participatory governance: village assemblies, planning, budgeting
  • Community infrastructure and assets
  • Social cohesion: belonging, exclusion, inter-group tension
  • Community-driven development programme outcomes
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES, individual_ids


def generate(n_individuals: int = 18000, seed: int = 807) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    ids = individual_ids(rng, n, prefix="COM")
    districts, urban = pick_districts(rng, n, urban_share=0.30)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.51, n)
    age = rng.integers(18, 70, n)
    educ_years = np.clip(rng.normal(6 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    wealth_quintile = rng.choice([1, 2, 3, 4, 5], n, p=[0.22, 0.21, 0.20, 0.19, 0.18])
    wealth = (wealth_quintile - 3) / 2.0 + rng.normal(0, 0.3, n)

    # --- Group memberships ---
    n_group_memberships = np.clip(
        rng.poisson(np.clip(1.2 + 0.3 * wealth + 0.2 * educ_years / 10 - 0.2 * female, 0.1, 10), n),
        0, 8
    )
    group_types_all = ["savings", "religious", "agricultural", "water_committee",
                       "health_committee", "youth", "women", "political"]
    # Assign primary group type for those with at least one membership
    group_types = np.where(
        n_group_memberships > 0,
        rng.choice(group_types_all, n, p=[0.18, 0.16, 0.15, 0.10, 0.10, 0.12, 0.10, 0.09]),
        "none"
    )

    # --- Trust indicators (1-5) ---
    base_trust = 0.1 * educ_years / 10 + 0.1 * wealth
    trust_neighbors = np.clip(
        rng.normal(3.5 + base_trust - 0.3 * urban.astype(float), 0.8, n), 1, 5
    ).round(0).astype(int)
    trust_strangers = np.clip(
        rng.normal(2.5 + base_trust - 0.4 * urban.astype(float), 0.9, n), 1, 5
    ).round(0).astype(int)
    trust_local_leaders = np.clip(
        rng.normal(3.0 + base_trust, 0.9, n), 1, 5
    ).round(0).astype(int)

    # --- Bonding vs bridging social capital ---
    bonding_social_capital_score = np.clip(
        rng.normal(0.5 + 0.05 * n_group_memberships + 0.1 * (trust_neighbors / 5)
                   - 0.05 * urban.astype(float), 0.15, n), 0, 1
    ).round(3)
    bridging_social_capital_score = np.clip(
        rng.normal(0.35 + 0.06 * n_group_memberships + 0.1 * (trust_strangers / 5)
                   + 0.08 * urban.astype(float) + 0.05 * educ_years / 10, 0.15, n), 0, 1
    ).round(3)

    # --- Collective action ---
    participated_in_collective_action = rng.binomial(
        1, _logistic(0.30, 0.1 * n_group_memberships + 0.1 * wealth + 0.05 * educ_years / 10, 0.4)
    )
    collective_action_type = np.where(
        participated_in_collective_action,
        rng.choice(["infrastructure", "natural_resource", "advocacy", "festival", "dispute_resolution"],
                   n, p=[0.28, 0.22, 0.18, 0.17, 0.15]),
        "none"
    )
    free_rider_perception = rng.binomial(
        1, _logistic(0.25, -trust_neighbors / 5 + 0.1 * urban.astype(float), 0.4)
    )

    # --- Participatory governance ---
    attended_village_assembly = rng.binomial(
        1, _logistic(0.35, 0.1 * n_group_memberships - 0.15 * female + 0.05 * educ_years / 10 - 0.1 * urban.astype(float), 0.3)
    )
    voiced_opinion_in_meeting = np.where(
        attended_village_assembly,
        rng.binomial(1, _logistic(0.40, 0.1 * educ_years / 10 + 0.1 * wealth - 0.2 * female, 0.4)),
        0
    )
    involved_in_planning = rng.binomial(
        1, _logistic(0.18, 0.1 * n_group_memberships + 0.1 * educ_years / 10 - 0.1 * female, 0.3)
    )
    aware_of_budget_allocation = rng.binomial(
        1, _logistic(0.22, 0.1 * educ_years / 10 + 0.1 * attended_village_assembly + 0.05 * urban.astype(float), 0.4)
    )

    # --- Community assets ---
    community_has_health_facility = rng.binomial(1, _logistic(0.55, 0.3 * urban.astype(float) + 0.1 * wealth, 0.3))
    community_has_school = rng.binomial(1, _logistic(0.65, 0.2 * urban.astype(float), 0.3))
    community_has_market = rng.binomial(1, _logistic(0.45, 0.3 * urban.astype(float) + 0.1 * wealth, 0.3))
    community_has_road = rng.binomial(1, _logistic(0.50, 0.35 * urban.astype(float) + 0.1 * wealth, 0.3))
    community_has_electricity = rng.binomial(1, _logistic(0.55, 0.35 * urban.astype(float) + 0.15 * wealth, 0.3))
    community_has_water_point = rng.binomial(1, _logistic(0.60, 0.2 * urban.astype(float) + 0.1 * wealth, 0.3))

    # --- Cohesion ---
    feels_belonging = np.clip(
        rng.normal(3.5 + 0.1 * n_group_memberships - 0.2 * urban.astype(float) + 0.1 * trust_neighbors / 5, 0.8, n),
        1, 5
    ).round(0).astype(int)
    experienced_social_exclusion = rng.binomial(
        1, _logistic(0.15, -wealth - 0.05 * educ_years / 10 + 0.1 * female, 0.3)
    )
    inter_group_tension = rng.binomial(1, _logistic(0.18, -trust_strangers / 5 + 0.1 * urban.astype(float), 0.3))
    willingness_to_help_neighbors = rng.binomial(
        1, _logistic(0.65, trust_neighbors / 5 + 0.05 * n_group_memberships - 0.1 * urban.astype(float), 0.3)
    )

    # --- Community-driven development ---
    in_cdd_programme = rng.binomial(1, 0.25, n)
    contributed_to_project = np.where(
        in_cdd_programme,
        rng.choice(["labor", "cash", "materials"], n, p=[0.45, 0.30, 0.25]),
        "none"
    )
    project_completed = np.where(
        in_cdd_programme,
        rng.binomial(1, 0.65 + 0.10 * (n_group_memberships > 1).astype(float)),
        0
    )
    satisfied_with_project = np.where(
        project_completed.astype(bool),
        rng.binomial(1, 0.60 + 0.10 * voiced_opinion_in_meeting),
        0
    )

    df = pd.DataFrame({
        "individual_id": ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "female": female, "age": age,
        "education_years": educ_years, "wealth_quintile": wealth_quintile,
        "n_group_memberships": n_group_memberships,
        "primary_group_type": group_types,
        "trust_neighbors": trust_neighbors,
        "trust_strangers": trust_strangers,
        "trust_local_leaders": trust_local_leaders,
        "bonding_social_capital_score": bonding_social_capital_score,
        "bridging_social_capital_score": bridging_social_capital_score,
        "participated_in_collective_action": participated_in_collective_action,
        "collective_action_type": collective_action_type,
        "free_rider_perception": free_rider_perception,
        "attended_village_assembly": attended_village_assembly,
        "voiced_opinion_in_meeting": voiced_opinion_in_meeting,
        "involved_in_planning": involved_in_planning,
        "aware_of_budget_allocation": aware_of_budget_allocation,
        "community_has_health_facility": community_has_health_facility,
        "community_has_school": community_has_school,
        "community_has_market": community_has_market,
        "community_has_road": community_has_road,
        "community_has_electricity": community_has_electricity,
        "community_has_water_point": community_has_water_point,
        "feels_belonging": feels_belonging,
        "experienced_social_exclusion": experienced_social_exclusion,
        "inter_group_tension": inter_group_tension,
        "willingness_to_help_neighbors": willingness_to_help_neighbors,
        "in_cdd_programme": in_cdd_programme,
        "contributed_to_project": contributed_to_project,
        "project_completed": project_completed,
        "satisfied_with_project": satisfied_with_project,
    })

    df = inject_missing(df,
        columns=["trust_neighbors", "trust_strangers", "bonding_social_capital_score", "education_years"],
        rates=[0.04, 0.05, 0.03, 0.03],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
