"""
Generator: Agriculture & Value Chain
─────────────────────────────────────
Simulates a value chain dataset tracking commodities from farm to market
with actors at each node: producers, aggregators, processors, and retailers.
Includes value addition, margins, quality grading, and contract farming.

Rows: ~25k transactions across the chain.

Realistic features:
  • Multi-node value chain (farm → aggregator → processor → retailer)
  • Value addition and margin calculations at each stage
  • Quality grading (A/B/C) affecting price
  • Contract vs. spot market sales
  • Post-harvest losses at each stage
  • Cooperative membership effects
  • Gender of actor at each node
"""

import numpy as np
import pandas as pd
from .utils import pick_districts, inject_missing, COUNTRIES


def generate(n_transactions: int = 25000, seed: int = 704) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = n_transactions

    ids = [f"TXN-{i:06d}" for i in range(1, n + 1)]
    districts, urban = pick_districts(rng, n, urban_share=0.20)
    countries = rng.choice(list(COUNTRIES.keys()), n)

    # Value chain stage
    chain_node = rng.choice(
        ["producer", "aggregator", "processor", "retailer"],
        n, p=[0.40, 0.20, 0.20, 0.20]
    )

    # Actor characteristics
    actor_female = rng.binomial(1, np.where(chain_node == "producer", 0.35,
                                  np.where(chain_node == "retailer", 0.45, 0.20)))
    actor_age = rng.integers(20, 65, n)
    actor_educ = np.clip(rng.normal(
        np.where(chain_node == "producer", 5,
        np.where(chain_node == "aggregator", 8,
        np.where(chain_node == "processor", 9, 7))), 3, n), 0, 18).astype(int)

    # Commodity
    commodity = rng.choice(
        ["maize", "coffee", "dairy", "poultry", "horticulture", "rice", "groundnuts", "cassava"],
        n, p=[0.18, 0.12, 0.10, 0.10, 0.15, 0.12, 0.12, 0.11]
    )

    # Cooperative/group membership
    in_cooperative = rng.binomial(1, np.where(chain_node == "producer", 0.35, 0.10))

    # Contract farming (producers)
    has_contract = np.where(chain_node == "producer",
                            rng.binomial(1, 0.20 + 0.15 * in_cooperative), 0)

    # Quality grade
    quality = rng.choice(["A", "B", "C"], n,
                         p=[0.25, 0.45, 0.30])
    quality_premium = np.where(quality == "A", 1.25, np.where(quality == "B", 1.0, 0.75))

    # Volume (kg)
    base_vol = np.where(chain_node == "producer", rng.lognormal(5, 0.8, n),
               np.where(chain_node == "aggregator", rng.lognormal(7, 0.6, n),
               np.where(chain_node == "processor", rng.lognormal(8, 0.5, n),
                        rng.lognormal(6, 0.7, n))))
    volume_kg = np.round(base_vol, 0)

    # Base price (USD/kg by commodity)
    base_prices = {"maize": 0.22, "coffee": 2.50, "dairy": 0.45, "poultry": 1.80,
                   "horticulture": 0.60, "rice": 0.40, "groundnuts": 0.65, "cassava": 0.12}
    base_price = np.array([base_prices[c] for c in commodity])

    # Price at each node (markup along chain)
    node_markup = np.where(chain_node == "producer", 1.0,
                  np.where(chain_node == "aggregator", 1.15,
                  np.where(chain_node == "processor", 1.45, 1.80)))
    contract_premium = np.where(has_contract, 1.08, 1.0)
    price_per_kg = np.round(base_price * node_markup * quality_premium * contract_premium
                            * rng.lognormal(0, 0.08, n), 3)

    # Revenue
    revenue_usd = np.round(price_per_kg * volume_kg, 2)

    # Costs
    input_cost_pct = np.where(chain_node == "producer", rng.uniform(0.30, 0.55, n),
                    np.where(chain_node == "aggregator", rng.uniform(0.05, 0.15, n),
                    np.where(chain_node == "processor", rng.uniform(0.15, 0.30, n),
                             rng.uniform(0.10, 0.25, n))))
    transport_cost_pct = rng.uniform(0.03, 0.15, n)
    total_cost = np.round(revenue_usd * (input_cost_pct + transport_cost_pct), 2)
    margin = np.round(revenue_usd - total_cost, 2)
    margin_pct = np.round((margin / np.maximum(revenue_usd, 1)) * 100, 1)

    # Post-harvest loss
    phl_pct = np.where(chain_node == "producer", rng.beta(2, 8, n) * 30,
              np.where(chain_node == "aggregator", rng.beta(2, 10, n) * 20,
              np.where(chain_node == "processor", rng.beta(1.5, 12, n) * 10,
                       rng.beta(1.5, 15, n) * 8)))
    phl_pct = np.round(phl_pct, 1)

    # Storage
    has_improved_storage = rng.binomial(1, np.where(chain_node == "producer", 0.15, 0.45))

    # Access to finance
    access_credit = rng.binomial(1, _logistic(0.25, 0.1 * actor_educ / 10 + 0.2 * in_cooperative, 0.4))

    # Certification (organic, fair trade)
    has_certification = rng.binomial(1, np.where(
        np.isin(commodity, ["coffee", "horticulture"]) & (chain_node == "producer"),
        0.15 + 0.15 * in_cooperative, 0.03))

    # Distance to market
    distance_market_km = np.clip(rng.exponential(
        np.where(chain_node == "producer", 15, np.where(chain_node == "aggregator", 8, 5))),
        0.5, 80).round(1)

    # Buyer type
    buyer_type = np.where(chain_node == "producer",
        rng.choice(["local_trader", "aggregator", "cooperative", "processor_direct", "export_agent"],
                   n, p=[0.35, 0.25, 0.20, 0.12, 0.08]),
        np.where(chain_node == "aggregator",
            rng.choice(["processor", "wholesale_market", "export_agent"], n, p=[0.40, 0.35, 0.25]),
            rng.choice(["retail_market", "supermarket", "restaurant", "export"], n, p=[0.35, 0.25, 0.20, 0.20])
        ))

    # Season
    season = rng.choice(["harvest", "post_harvest", "lean_season", "planting"],
                        n, p=[0.30, 0.25, 0.25, 0.20])

    df = pd.DataFrame({
        "transaction_id": ids, "country": countries, "district": districts,
        "chain_node": chain_node, "commodity": commodity, "quality_grade": quality,
        "actor_female": actor_female, "actor_age": actor_age,
        "actor_education_years": actor_educ,
        "in_cooperative": in_cooperative, "has_contract": has_contract,
        "has_certification": has_certification,
        "volume_kg": volume_kg, "price_per_kg_usd": price_per_kg,
        "revenue_usd": revenue_usd, "total_cost_usd": total_cost,
        "margin_usd": margin, "margin_pct": margin_pct,
        "post_harvest_loss_pct": phl_pct,
        "has_improved_storage": has_improved_storage,
        "access_credit": access_credit,
        "distance_to_market_km": distance_market_km,
        "buyer_type": buyer_type, "season": season,
        "transport_cost_pct": np.round(transport_cost_pct * 100, 1),
    })

    df = inject_missing(df,
        columns=["margin_pct", "post_harvest_loss_pct", "price_per_kg_usd", "distance_to_market_km"],
        rates=[0.05, 0.06, 0.03, 0.04],
        rng=rng, mechanism="MCAR")
    return df


def _logistic(base, z, slope=1.0):
    from scipy.special import expit
    return expit(np.log(np.clip(base, 1e-6, 1-1e-6) / (1 - np.clip(base, 1e-6, 1-1e-6))) + slope * np.asarray(z))
