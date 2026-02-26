"""
Generator: Advocacy, Rights & Legal Empowerment
────────────────────────────────────────────────
Simulates data from a legal aid / rights-based programme with case records,
awareness of rights, access to justice, legal identity, land tenure, and
advocacy campaign tracking.

Rows: ~15k individuals.

Realistic features:
  • Legal identity (birth registration, national ID)
  • Land tenure security and documentation
  • Access to justice (formal courts, customary, legal aid)
  • Awareness of rights (CEDAW, child rights, labor rights)
  • Dispute types and resolution mechanisms
  • Advocacy campaign reach and behavior change
  • Freedom of expression / civic space indicators
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_individuals: int = 15000, seed: int = 708) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_individuals

    ids = [f"ADV-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.35)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    female = rng.binomial(1, 0.52, n)
    age = rng.integers(15, 70, n)
    educ_years = np.clip(rng.normal(6 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)
    wealth = rng.normal(0, 1, n) + 0.3 * urban.astype(float)

    # Legal identity
    has_birth_certificate = rng.binomial(1, _logistic(0.55, wealth + 0.1 * urban.astype(float), 0.4))
    has_national_id = rng.binomial(1, _logistic(0.60, wealth + 0.15 * urban.astype(float) + 0.05 * (age > 18).astype(float), 0.4))

    # Land tenure
    owns_land = rng.binomial(1, _logistic(0.35, wealth - 0.1 * female, 0.3))
    has_land_title = np.where(owns_land, rng.binomial(1, _logistic(0.25, wealth + 0.1 * urban.astype(float), 0.4)), 0)
    land_dispute_experienced = np.where(owns_land, rng.binomial(1, 0.15), 0)

    # Programme participation
    received_legal_aid = rng.binomial(1, 0.18, n)
    attended_rights_training = rng.binomial(1, 0.22, n)
    in_advocacy_group = rng.binomial(1, 0.15, n)

    # Awareness of rights (0-10 scale)
    rights_awareness = np.clip(
        rng.normal(4 + 0.8 * educ_years / 10 + 0.5 * attended_rights_training
                   + 0.3 * urban.astype(float), 1.5, n), 0, 10
    ).round(1)

    knows_cedaw = rng.binomial(1, _logistic(0.20, rights_awareness / 10, 0.8))
    knows_child_rights = rng.binomial(1, _logistic(0.35, rights_awareness / 10, 0.7))
    knows_labor_rights = rng.binomial(1, _logistic(0.30, rights_awareness / 10, 0.7))
    knows_land_rights = rng.binomial(1, _logistic(0.28, rights_awareness / 10 + 0.1 * owns_land, 0.6))

    # Dispute experience and resolution
    experienced_dispute = rng.binomial(1, 0.22, n)
    dispute_type = np.where(experienced_dispute,
        rng.choice(["land", "family_inheritance", "labor", "domestic_violence", "property",
                    "debt", "criminal", "other"],
                   n, p=[0.20, 0.15, 0.15, 0.12, 0.12, 0.10, 0.08, 0.08]),
        "none")
    sought_resolution = np.where(experienced_dispute,
        rng.binomial(1, _logistic(0.55, wealth + 0.2 * received_legal_aid + 0.1 * rights_awareness / 10, 0.4)), 0)
    resolution_mechanism = np.where(sought_resolution.astype(bool),
        rng.choice(["formal_court", "customary_leader", "legal_aid_clinic", "mediation",
                    "police", "human_rights_commission", "self_resolved"],
                   n, p=[0.18, 0.25, 0.15, 0.15, 0.12, 0.05, 0.10]),
        "none")
    dispute_resolved = np.where(sought_resolution.astype(bool),
        rng.binomial(1, 0.55 + 0.10 * received_legal_aid), 0)
    satisfied_with_outcome = np.where(dispute_resolved.astype(bool),
        rng.binomial(1, 0.60), 0)

    # Access to justice barriers
    barrier_cost = np.where(experienced_dispute & ~sought_resolution.astype(bool),
        rng.binomial(1, 0.40), 0)
    barrier_distance = np.where(experienced_dispute & ~sought_resolution.astype(bool),
        rng.binomial(1, 0.25), 0)
    barrier_fear = np.where(experienced_dispute & ~sought_resolution.astype(bool),
        rng.binomial(1, 0.30), 0)
    barrier_distrust = np.where(experienced_dispute & ~sought_resolution.astype(bool),
        rng.binomial(1, 0.20), 0)

    # Civic participation
    voted_last_election = rng.binomial(1, _logistic(0.55, 0.05 * (age > 18).astype(float) + 0.1 * educ_years / 10, 0.3))
    attended_community_meeting = rng.binomial(1, _logistic(0.35, 0.1 * in_advocacy_group + 0.05 * educ_years / 10, 0.3))
    feels_can_influence_decisions = rng.binomial(1, _logistic(0.30, wealth + 0.2 * in_advocacy_group + 0.1 * rights_awareness / 10, 0.3))

    # Freedom / civic space perception (1-5)
    civic_space_perception = np.clip(
        rng.normal(3 + 0.2 * urban.astype(float) + 0.1 * rights_awareness / 10, 0.8, n), 1, 5
    ).round(1)

    # Advocacy campaign exposure
    exposed_to_campaign = rng.binomial(1, 0.35 + 0.10 * urban.astype(float))
    campaign_channel = np.where(exposed_to_campaign,
        rng.choice(["radio", "community_meeting", "social_media", "poster_leaflet", "tv", "door_to_door"],
                   n, p=[0.25, 0.22, 0.18, 0.15, 0.10, 0.10]),
        "none")
    changed_behavior_after_campaign = np.where(exposed_to_campaign,
        rng.binomial(1, 0.25 + 0.10 * attended_rights_training), 0)

    df = pd.DataFrame({
        "individual_id": ids, "country": countries, "district": districts,
        "urban": urban.astype(int), "female": female, "age": age,
        "education_years": educ_years,
        "has_birth_certificate": has_birth_certificate,
        "has_national_id": has_national_id,
        "owns_land": owns_land, "has_land_title": has_land_title,
        "land_dispute_experienced": land_dispute_experienced,
        "received_legal_aid": received_legal_aid,
        "attended_rights_training": attended_rights_training,
        "in_advocacy_group": in_advocacy_group,
        "rights_awareness_score": rights_awareness,
        "knows_cedaw": knows_cedaw, "knows_child_rights": knows_child_rights,
        "knows_labor_rights": knows_labor_rights, "knows_land_rights": knows_land_rights,
        "experienced_dispute": experienced_dispute, "dispute_type": dispute_type,
        "sought_resolution": sought_resolution,
        "resolution_mechanism": resolution_mechanism,
        "dispute_resolved": dispute_resolved,
        "satisfied_with_outcome": satisfied_with_outcome,
        "barrier_cost": barrier_cost, "barrier_distance": barrier_distance,
        "barrier_fear": barrier_fear, "barrier_distrust": barrier_distrust,
        "voted_last_election": voted_last_election,
        "attended_community_meeting": attended_community_meeting,
        "feels_can_influence_decisions": feels_can_influence_decisions,
        "civic_space_perception": civic_space_perception,
        "exposed_to_advocacy_campaign": exposed_to_campaign,
        "campaign_channel": campaign_channel,
        "changed_behavior_post_campaign": changed_behavior_after_campaign,
    })

    df = inject_missing(df,
        columns=["rights_awareness_score", "civic_space_perception", "education_years"],
        rates=[0.04, 0.05, 0.03],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
