"""
Generator: Water, Sanitation & Hygiene (WASH)
──────────────────────────────────────────────
Simulates a household-level WASH monitoring dataset aligned with JMP
(WHO/UNICEF Joint Monitoring Programme) service ladders, water quality
testing results, hygiene observation protocols, CLTS programme tracking,
child diarrhoea outcomes, and school-WASH indicators.

Rows: one per household (~18k).

Realistic features:
  • JMP service ladders for drinking water (safely managed → surface water)
    and sanitation (safely managed → open defecation)
  • Water quality testing — E. coli (CFU/100 mL) and turbidity (NTU)
    correlated with source type and treatment
  • Water quantity (litres/person/day), round-trip collection time, and
    queuing time — meeting Sphere minimum (15 L/p/d)
  • Handwashing observation (water + soap present at fixed place) versus
    self-reported practice at critical times
  • Menstrual Hygiene Management (MHM) facility availability
  • CLTS (Community-Led Total Sanitation) triggering, ODF declaration and
    verification status
  • Child diarrhoea (under-5, last 2 weeks) prevalence linked to WASH
    conditions via logistic model
  • School WASH module: water point, sex-separated toilets, handwashing
    station, MHM facilities at schools
  • Household water treatment practices
"""

import numpy as np
import pandas as pd
from .utils import household_ids, pick_districts, inject_missing, COUNTRIES


def generate(n_households: int = 18000, seed: int = 710) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_households

    # --- IDs & geography ---
    hh_ids = household_ids(rng, n, prefix="WSH")
    districts, urban = pick_districts(rng, n, urban_share=0.28)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # --- Household demographics ---
    hh_size = rng.choice(range(1, 11), n,
                         p=[0.02, 0.05, 0.09, 0.14, 0.22, 0.20, 0.13, 0.08, 0.05, 0.02])
    n_children_under5 = np.clip(rng.poisson(0.8, n), 0, 5)
    has_children_u5 = (n_children_under5 > 0).astype(int)
    head_female = rng.binomial(1, 0.28, n)
    head_educ = np.clip(rng.normal(5.5 + 2 * urban.astype(float), 3.5, n), 0, 18).astype(int)

    # Latent wealth
    wealth = rng.normal(0, 1, n) + 0.5 * urban.astype(float) + 0.12 * head_educ / 10
    wealth_quintile = pd.qcut(wealth, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Programme participation
    in_wash_programme = rng.binomial(1, 0.30, n)

    # ------------------------------------------------------------------ #
    # WATER SUPPLY — JMP service ladder
    # ------------------------------------------------------------------ #
    water_source_options = [
        "piped_dwelling", "piped_yard", "public_tap", "borehole",
        "protected_well", "protected_spring", "unprotected_well",
        "unprotected_spring", "surface_water", "rainwater", "tanker_truck",
    ]
    # Probabilities shift with wealth and urban
    wp = np.column_stack([
        _logistic(0.06, 0.5 * wealth + 0.6 * urban.astype(float), 0.5),  # piped dwelling
        _logistic(0.06, 0.35 * wealth + 0.3 * urban.astype(float), 0.4), # piped yard
        _logistic(0.10, 0.15 * urban.astype(float), 0.2),                # public tap
        _logistic(0.18, 0.08 * wealth, 0.2),                             # borehole
        _logistic(0.10, -0.05 * wealth, 0.2),                            # protected well
        _logistic(0.08, -0.08 * urban.astype(float), 0.2),               # protected spring
        _logistic(0.09, -0.15 * wealth - 0.10 * urban.astype(float), 0.3),  # unprotected well
        _logistic(0.08, -0.15 * wealth - 0.08 * urban.astype(float), 0.3),  # unprotected spring
        _logistic(0.08, -0.30 * wealth - 0.20 * urban.astype(float), 0.4),  # surface water
        _logistic(0.05, 0, 0.1) * np.ones(n),                            # rainwater
        _logistic(0.04, 0.08 * urban.astype(float), 0.2),                # tanker truck
    ])
    wp = wp / wp.sum(axis=1, keepdims=True)
    water_source = np.array([rng.choice(water_source_options, p=wp[i]) for i in range(n)])

    improved_water_set = {"piped_dwelling", "piped_yard", "public_tap", "borehole",
                          "protected_well", "protected_spring", "rainwater"}
    water_improved = np.array([1 if s in improved_water_set else 0 for s in water_source])

    # Water on premises
    water_on_premises = np.isin(water_source, ["piped_dwelling", "piped_yard"]).astype(int)

    # Round-trip collection time (minutes)
    water_time_roundtrip_min = np.where(
        water_on_premises, 0,
        np.clip(rng.exponential(
            np.where(urban, 10, 25) - 3 * water_improved
        ), 1, 120).astype(int),
    )

    # Queuing time (minutes, if not on premises)
    queuing_time_min = np.where(
        water_on_premises, 0,
        np.clip(rng.exponential(np.where(urban, 8, 15)), 0, 90).astype(int),
    )

    # JMP water service level
    jmp_water = np.where(
        water_on_premises & water_improved, "safely_managed",
        np.where(water_improved & (water_time_roundtrip_min <= 30), "basic",
        np.where(water_improved, "limited",
        np.where(water_source == "surface_water", "surface_water", "unimproved")))
    )

    # ------------------------------------------------------------------ #
    # Water quality testing
    # ------------------------------------------------------------------ #
    # E. coli (CFU/100 mL) — log-normal, correlated with source type
    ecoli_base_mu = np.where(
        jmp_water == "safely_managed", 0.5,
        np.where(jmp_water == "basic", 2.0,
        np.where(jmp_water == "limited", 2.5, 3.8))
    )
    ecoli_cfu_100ml = np.round(
        np.clip(rng.lognormal(ecoli_base_mu + rng.normal(0, 0.4, n), 1.2, n), 0, 5000), 0
    ).astype(int)
    ecoli_risk_category = np.where(
        ecoli_cfu_100ml == 0, "conformity",
        np.where(ecoli_cfu_100ml <= 10, "low",
        np.where(ecoli_cfu_100ml <= 100, "intermediate",
        np.where(ecoli_cfu_100ml <= 1000, "high", "very_high")))
    )

    # Turbidity (NTU)
    turbidity_ntu = np.round(
        np.clip(rng.lognormal(np.where(water_improved, 0.5, 1.8), 0.8, n), 0.1, 500), 1
    )

    # Free chlorine residual (mg/L, only for treated piped/tap)
    chlorine_tested = np.isin(water_source, ["piped_dwelling", "piped_yard", "public_tap"]).astype(int)
    free_chlorine_residual = np.where(
        chlorine_tested,
        np.round(np.clip(rng.normal(0.3, 0.2, n), 0, 2.0), 2),
        np.nan,
    )

    # Water quantity (litres/person/day)
    liters_per_person_day = np.round(np.clip(
        rng.normal(
            np.where(water_on_premises, 50, 18)
            + 5 * wealth + 5 * urban.astype(float)
            + 3 * in_wash_programme,
            10, n),
        2, 150), 1)
    sufficient_water_15lpd = (liters_per_person_day >= 15).astype(int)  # Sphere minimum
    sufficient_water_20lpd = (liters_per_person_day >= 20).astype(int)  # WHO basic

    # ------------------------------------------------------------------ #
    # Household water treatment
    # ------------------------------------------------------------------ #
    treats_water = rng.binomial(1, _logistic(
        0.25, 0.12 * wealth + 0.15 * in_wash_programme
        + 0.08 * head_educ / 10 + 0.10 * (1 - water_improved), 0.3
    ))
    treatment_method = np.where(
        treats_water,
        rng.choice(["boiling", "chlorination", "solar_disinfection",
                     "ceramic_filter", "biosand_filter", "cloth_filter", "other"],
                   n, p=[0.28, 0.25, 0.10, 0.12, 0.10, 0.08, 0.07]),
        "none",
    )

    # ------------------------------------------------------------------ #
    # SANITATION — JMP service ladder
    # ------------------------------------------------------------------ #
    sanitation_options = [
        "flush_sewer", "flush_septic", "pit_latrine_vip",
        "pit_latrine_slab", "pit_latrine_no_slab",
        "composting_toilet", "bucket_latrine", "hanging_toilet",
        "open_defecation",
    ]
    sp = np.column_stack([
        _logistic(0.04, 0.45 * wealth + 0.5 * urban.astype(float), 0.5),   # flush sewer
        _logistic(0.07, 0.30 * wealth + 0.20 * urban.astype(float), 0.4),  # flush septic
        _logistic(0.06, 0.15 * wealth, 0.3),                                # VIP latrine
        _logistic(0.18, 0.10 * wealth, 0.2),                                # pit w/ slab
        _logistic(0.18, -0.12 * wealth - 0.05 * urban.astype(float), 0.3), # pit no slab
        _logistic(0.03, 0, 0.1) * np.ones(n),                               # composting
        _logistic(0.03, -0.10 * wealth, 0.2),                               # bucket
        _logistic(0.04, -0.12 * wealth - 0.08 * urban.astype(float), 0.2), # hanging
        _logistic(0.12, -0.35 * wealth - 0.30 * urban.astype(float), 0.5), # open defecation
    ])
    sp = sp / sp.sum(axis=1, keepdims=True)
    sanitation_facility = np.array([rng.choice(sanitation_options, p=sp[i]) for i in range(n)])

    improved_sanitation_set = {"flush_sewer", "flush_septic", "pit_latrine_vip",
                               "pit_latrine_slab", "composting_toilet"}
    sanitation_improved = np.array([1 if s in improved_sanitation_set else 0
                                    for s in sanitation_facility])

    open_defecation = (sanitation_facility == "open_defecation").astype(int)

    # Shared facility
    shared_facility = np.where(
        ~open_defecation.astype(bool),
        rng.binomial(1, _logistic(0.25, -0.10 * wealth - 0.05 * urban.astype(float), 0.2)),
        0,
    )

    # JMP sanitation service level
    jmp_sanitation = np.where(
        np.isin(sanitation_facility, ["flush_sewer"]) & (shared_facility == 0),
        "safely_managed",
        np.where(sanitation_improved & (shared_facility == 0), "basic",
        np.where(sanitation_improved, "limited",
        np.where(open_defecation, "open_defecation", "unimproved")))
    )

    # Excreta disposal (for pit latrines — containment)
    pit_latrine_mask = np.isin(sanitation_facility,
                               ["pit_latrine_vip", "pit_latrine_slab", "pit_latrine_no_slab"])
    pit_emptied_safely = np.where(
        pit_latrine_mask,
        rng.binomial(1, _logistic(0.30, 0.15 * wealth + 0.1 * urban.astype(float), 0.3)),
        -1,  # not applicable
    )

    # ------------------------------------------------------------------ #
    # CLTS (Community-Led Total Sanitation)
    # ------------------------------------------------------------------ #
    clts_triggered = rng.binomial(1, _logistic(
        0.28, -0.10 * urban.astype(float) + 0.08 * in_wash_programme, 0.2
    ))
    community_odf_declared = np.where(
        clts_triggered,
        rng.binomial(1, _logistic(0.40, 0.10 * wealth + 0.12 * in_wash_programme, 0.3)),
        0,
    )
    community_odf_verified = np.where(
        community_odf_declared,
        rng.binomial(1, 0.60, n),
        0,
    )
    # Slippage: verified communities where OD resumes
    odf_slippage = np.where(
        community_odf_verified,
        rng.binomial(1, 0.15, n),
        0,
    )

    # ------------------------------------------------------------------ #
    # HYGIENE — observation + reported
    # ------------------------------------------------------------------ #
    # Handwashing observation (enumerator checks fixed handwashing place)
    hw_place_observed = rng.binomial(1, _logistic(
        0.40, 0.20 * wealth + 0.15 * in_wash_programme + 0.10 * urban.astype(float), 0.3
    ))
    hw_water_present = np.where(hw_place_observed, rng.binomial(1, 0.78, n), 0)
    hw_soap_present = np.where(hw_place_observed,
                               rng.binomial(1, _logistic(0.50, 0.15 * wealth + 0.10 * in_wash_programme, 0.3)), 0)
    hw_water_and_soap = (hw_water_present.astype(bool) & hw_soap_present.astype(bool)).astype(int)

    # JMP hygiene service level
    hygiene_service_level = np.where(
        hw_water_and_soap, "basic",
        np.where(hw_place_observed, "limited", "no_facility")
    )

    # Reported handwashing at critical times (always higher than observed)
    hw_after_toilet_reported = rng.binomial(1, _logistic(0.70, 0.08 * head_educ / 10, 0.2))
    hw_before_eating_reported = rng.binomial(1, _logistic(0.65, 0.08 * head_educ / 10, 0.2))
    hw_before_cooking_reported = rng.binomial(1, _logistic(0.48, 0.08 * head_educ / 10, 0.2))
    hw_after_child_cleaning_reported = rng.binomial(1, _logistic(0.40, 0.05 * head_educ / 10, 0.2))

    # ------------------------------------------------------------------ #
    # Menstrual Hygiene Management (MHM)
    # ------------------------------------------------------------------ #
    mhm_private_space = rng.binomial(1, _logistic(
        0.50, 0.18 * wealth + 0.12 * urban.astype(float), 0.3
    ))
    mhm_materials_available = rng.binomial(1, _logistic(
        0.42, 0.20 * wealth + 0.10 * in_wash_programme, 0.3
    ))
    mhm_disposal_facility = rng.binomial(1, _logistic(
        0.30, 0.15 * wealth + 0.10 * urban.astype(float), 0.3
    ))

    # ------------------------------------------------------------------ #
    # Child diarrhoea (under-5, last 2 weeks)
    # ------------------------------------------------------------------ #
    diarrhea_risk_logit = (
        -1.5
        - 0.40 * water_improved
        - 0.35 * sanitation_improved
        - 0.30 * hw_water_and_soap
        + 0.35 * open_defecation
        + 0.002 * np.clip(ecoli_cfu_100ml, 0, 500) / 100
        - 0.20 * treats_water
        - 0.15 * wealth
        + rng.normal(0, 0.3, n)
    )
    diarrhea_prob = _logistic(0.15, diarrhea_risk_logit, 0.5)
    child_diarrhea_2wk = np.where(
        has_children_u5, rng.binomial(1, diarrhea_prob), -1  # -1 = no child u5
    )

    # Treatment seeking and ORS/zinc
    diarrhea_sought_treatment = np.where(
        child_diarrhea_2wk == 1,
        rng.binomial(1, _logistic(0.55, 0.20 * wealth + 0.12 * urban.astype(float), 0.3)),
        0,
    )
    diarrhea_ors_used = np.where(
        child_diarrhea_2wk == 1,
        rng.binomial(1, _logistic(0.40, 0.12 * wealth + 0.08 * head_educ / 10, 0.3)),
        0,
    )
    diarrhea_zinc_used = np.where(
        child_diarrhea_2wk == 1,
        rng.binomial(1, _logistic(0.18, 0.10 * wealth + 0.08 * head_educ / 10, 0.3)),
        0,
    )

    # ------------------------------------------------------------------ #
    # School WASH indicators (households with school-age children 5-17)
    # ------------------------------------------------------------------ #
    n_children_school_age = np.clip(rng.poisson(1.2 + 0.08 * hh_size), 0, 6)
    has_school_age = (n_children_school_age > 0).astype(int)

    school_has_water_point = np.where(
        has_school_age,
        rng.binomial(1, _logistic(0.55, 0.15 * urban.astype(float) + 0.08 * wealth, 0.3)),
        -1,
    )
    school_has_toilets = np.where(
        has_school_age,
        rng.binomial(1, _logistic(0.62, 0.12 * urban.astype(float) + 0.06 * wealth, 0.3)),
        -1,
    )
    school_separate_toilets_girls = np.where(
        (has_school_age == 1) & (np.array(school_has_toilets) == 1),
        rng.binomial(1, _logistic(0.48, 0.10 * urban.astype(float) + 0.05 * wealth, 0.2)),
        -1,
    )
    school_has_handwashing = np.where(
        has_school_age,
        rng.binomial(1, _logistic(0.35, 0.12 * urban.astype(float) + 0.06 * wealth, 0.3)),
        -1,
    )
    school_has_mhm_facility = np.where(
        has_school_age,
        rng.binomial(1, _logistic(0.28, 0.10 * urban.astype(float) + 0.05 * wealth, 0.3)),
        -1,
    )
    school_pupil_toilet_ratio = np.where(
        (has_school_age == 1) & (np.array(school_has_toilets) == 1),
        np.clip(rng.normal(np.where(urban, 40, 65), 20, n), 10, 200).astype(int),
        -1,
    )

    # ------------------------------------------------------------------ #
    # Assemble DataFrame
    # ------------------------------------------------------------------ #
    df = pd.DataFrame({
        "household_id": hh_ids,
        "country": countries,
        "district": districts,
        "urban": urban.astype(int),
        "household_size": hh_size,
        "n_children_under5": n_children_under5,
        "head_female": head_female,
        "head_education_years": head_educ,
        "wealth_quintile": wealth_quintile,
        "in_wash_programme": in_wash_programme,
        # Water supply
        "water_source": water_source,
        "water_improved": water_improved,
        "water_on_premises": water_on_premises,
        "water_time_roundtrip_min": water_time_roundtrip_min,
        "queuing_time_min": queuing_time_min,
        "jmp_water_service_level": jmp_water,
        # Water quality
        "ecoli_cfu_100ml": ecoli_cfu_100ml,
        "ecoli_risk_category": ecoli_risk_category,
        "turbidity_ntu": turbidity_ntu,
        "free_chlorine_residual_mg_l": free_chlorine_residual,
        # Water quantity
        "liters_per_person_day": liters_per_person_day,
        "sufficient_water_15lpd": sufficient_water_15lpd,
        "sufficient_water_20lpd": sufficient_water_20lpd,
        # Household water treatment
        "treats_water": treats_water,
        "water_treatment_method": treatment_method,
        # Sanitation
        "sanitation_facility": sanitation_facility,
        "sanitation_improved": sanitation_improved,
        "open_defecation": open_defecation,
        "shared_sanitation_facility": shared_facility,
        "jmp_sanitation_service_level": jmp_sanitation,
        "pit_emptied_safely": pit_emptied_safely,
        # CLTS
        "clts_triggered": clts_triggered,
        "community_odf_declared": community_odf_declared,
        "community_odf_verified": community_odf_verified,
        "odf_slippage": odf_slippage,
        # Hygiene — observed
        "hw_place_observed": hw_place_observed,
        "hw_water_present": hw_water_present,
        "hw_soap_present": hw_soap_present,
        "hw_water_and_soap": hw_water_and_soap,
        "hygiene_service_level": hygiene_service_level,
        # Hygiene — reported
        "hw_after_toilet_reported": hw_after_toilet_reported,
        "hw_before_eating_reported": hw_before_eating_reported,
        "hw_before_cooking_reported": hw_before_cooking_reported,
        "hw_after_child_cleaning_reported": hw_after_child_cleaning_reported,
        # MHM
        "mhm_private_space": mhm_private_space,
        "mhm_materials_available": mhm_materials_available,
        "mhm_disposal_facility": mhm_disposal_facility,
        # Child diarrhoea
        "child_diarrhea_2wk": child_diarrhea_2wk,
        "diarrhea_sought_treatment": diarrhea_sought_treatment,
        "diarrhea_ors_used": diarrhea_ors_used,
        "diarrhea_zinc_used": diarrhea_zinc_used,
        # School WASH
        "has_school_age_children": has_school_age,
        "school_has_water_point": school_has_water_point,
        "school_has_toilets": school_has_toilets,
        "school_separate_toilets_girls": school_separate_toilets_girls,
        "school_has_handwashing": school_has_handwashing,
        "school_has_mhm_facility": school_has_mhm_facility,
        "school_pupil_toilet_ratio": school_pupil_toilet_ratio,
    })

    # Inject realistic missingness
    df = inject_missing(
        df,
        columns=[
            "ecoli_cfu_100ml", "turbidity_ntu", "liters_per_person_day",
            "water_time_roundtrip_min", "hw_place_observed",
            "child_diarrhea_2wk", "school_has_water_point",
            "school_pupil_toilet_ratio",
        ],
        rates=[0.12, 0.12, 0.06, 0.04, 0.05, 0.03, 0.08, 0.10],
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
