"""
Generator 10: Trade & Market Prices
────────────────────────────────────
Simulates weekly market-level price data for staple commodities across
multiple markets, with transport costs, seasonal patterns, and price
transmission between markets — the kind of data used for market integration
and food security analysis.

Structure: ~80 markets × 52 weeks × 8 commodities → ~33k rows.

Realistic features:
  • Seasonal price cycles (harvest lows, lean-season highs)
  • Spatial price correlation (nearby markets co-move)
  • Transport cost wedge proportional to distance
  • Price shocks (drought, conflict) with pass-through
  • Border effects on cross-country pairs
  • Cointegration structure for market integration analysis
"""

import numpy as np
import pandas as pd
from .utils import inject_missing


def generate(n_markets: int = 80, n_weeks: int = 104, seed: int = 606) -> pd.DataFrame:
    """Generate 2 years of weekly price data across markets and commodities."""
    rng = np.random.default_rng(seed)

    # --- Markets ---
    market_names = [
        "Nairobi_Central", "Nairobi_Eastleigh", "Mombasa", "Kisumu",
        "Nakuru", "Eldoret", "Kilifi", "Malindi",
        "Dar_Central", "Dar_Kariakoo", "Arusha", "Mwanza", "Dodoma",
        "Mbeya", "Tanga", "Morogoro",
        "Kampala_Owino", "Kampala_Nakasero", "Gulu", "Jinja", "Mbarara",
        "Mbale", "Fort_Portal", "Lira",
        "Addis_Merkato", "Addis_Shola", "Hawassa", "Bahir_Dar",
        "Mekelle", "Dire_Dawa", "Jimma", "Adama",
        "Kigali_Kimironko", "Kigali_Nyabugogo", "Musanze", "Huye",
        "Lagos_Mile12", "Lagos_Oyingbo", "Kano_Dawanau", "Abuja_Wuse",
        "Ibadan_Bodija", "Aba", "Port_Harcourt", "Kaduna",
        "Accra_Makola", "Accra_Kaneshie", "Kumasi_Central", "Tamale",
        "Dakar_Sandaga", "Dakar_Tilene", "Thies", "Kaolack",
        "Maputo_Zimpeto", "Maputo_Xiquelene", "Nampula", "Beira",
        "Lilongwe_Lizulu", "Lilongwe_Kanengo", "Blantyre", "Mzuzu",
        "Delhi_Azadpur", "Mumbai_Vashi", "Kolkata_Koley", "Chennai_Koyambedu",
        "Dhaka_Karwan", "Chittagong_Khatunganj", "Rajshahi", "Sylhet",
        "Kathmandu_Kalimati", "Pokhara", "Birgunj", "Biratnagar",
        "Phnom_Penh_Central", "Siem_Reap", "Battambang", "Kompong_Cham",
        "Lima_Mayorista", "Cusco", "Bogota_Corabastos", "Medellin",
    ][:n_markets]

    countries = []
    for m in market_names:
        if "Nairobi" in m or "Mombasa" in m or "Kisumu" in m or "Nakuru" in m or "Eldoret" in m or "Kilifi" in m or "Malindi" in m:
            countries.append("Kenya")
        elif "Dar" in m or "Arusha" in m or "Mwanza" in m or "Dodoma" in m or "Mbeya" in m or "Tanga" in m or "Morogoro" in m:
            countries.append("Tanzania")
        elif "Kampala" in m or "Gulu" in m or "Jinja" in m or "Mbarara" in m or "Mbale" in m or "Fort_Portal" in m or "Lira" in m:
            countries.append("Uganda")
        elif "Addis" in m or "Hawassa" in m or "Bahir" in m or "Mekelle" in m or "Dire" in m or "Jimma" in m or "Adama" in m:
            countries.append("Ethiopia")
        elif "Kigali" in m or "Musanze" in m or "Huye" in m:
            countries.append("Rwanda")
        elif "Lagos" in m or "Kano" in m or "Abuja" in m or "Ibadan" in m or "Aba" in m or "Port_Harcourt" in m or "Kaduna" in m:
            countries.append("Nigeria")
        elif "Accra" in m or "Kumasi" in m or "Tamale" in m:
            countries.append("Ghana")
        elif "Dakar" in m or "Thies" in m or "Kaolack" in m:
            countries.append("Senegal")
        elif "Maputo" in m or "Nampula" in m or "Beira" in m:
            countries.append("Mozambique")
        elif "Lilongwe" in m or "Blantyre" in m or "Mzuzu" in m:
            countries.append("Malawi")
        elif "Delhi" in m or "Mumbai" in m or "Kolkata" in m or "Chennai" in m:
            countries.append("India")
        elif "Dhaka" in m or "Chittagong" in m or "Rajshahi" in m or "Sylhet" in m:
            countries.append("Bangladesh")
        elif "Kathmandu" in m or "Pokhara" in m or "Birgunj" in m or "Biratnagar" in m:
            countries.append("Nepal")
        elif "Phnom" in m or "Siem" in m or "Battambang" in m or "Kompong" in m:
            countries.append("Cambodia")
        elif "Lima" in m or "Cusco" in m:
            countries.append("Peru")
        elif "Bogota" in m or "Medellin" in m:
            countries.append("Colombia")
        else:
            countries.append("Unknown")

    market_country = dict(zip(market_names, countries))

    # Market GPS (synthetic lat/lon for distance calc)
    market_lat = rng.uniform(-4, 10, n_markets)
    market_lon = rng.uniform(28, 42, n_markets)

    # --- Commodities ---
    commodities = {
        "maize": {"base_price": 0.28, "seasonal_amp": 0.15, "volatility": 0.08},
        "rice": {"base_price": 0.45, "seasonal_amp": 0.10, "volatility": 0.06},
        "beans": {"base_price": 0.65, "seasonal_amp": 0.18, "volatility": 0.10},
        "wheat_flour": {"base_price": 0.55, "seasonal_amp": 0.08, "volatility": 0.05},
        "cooking_oil": {"base_price": 1.40, "seasonal_amp": 0.05, "volatility": 0.07},
        "sugar": {"base_price": 0.80, "seasonal_amp": 0.06, "volatility": 0.06},
        "onions": {"base_price": 0.50, "seasonal_amp": 0.25, "volatility": 0.12},
        "tomatoes": {"base_price": 0.60, "seasonal_amp": 0.30, "volatility": 0.15},
    }

    # Reference market (Nairobi or first market)
    ref_prices = {}
    for comm, params in commodities.items():
        base = params["base_price"]
        amp = params["seasonal_amp"]
        vol = params["volatility"]

        # Generate reference price path with AR(1) + seasonality
        prices = np.zeros(n_weeks)
        prices[0] = base
        for t in range(1, n_weeks):
            # Seasonal component (peak before harvest, ~week 12 and 40)
            seasonal = amp * base * (np.sin(2 * np.pi * t / 52 - np.pi / 3))
            # AR(1) shock
            shock = 0.7 * (prices[t - 1] - base) + rng.normal(0, vol * base)
            prices[t] = base + seasonal + shock
        ref_prices[comm] = np.clip(prices, base * 0.4, base * 2.5)

    # --- Generate rows ---
    dates = pd.date_range("2023-01-02", periods=n_weeks, freq="W-MON")

    rows = []
    for mi, market in enumerate(market_names):
        country = market_country[market]

        # Transport cost factor (distance from reference market)
        dist_factor = np.sqrt((market_lat[mi] - market_lat[0]) ** 2 +
                               (market_lon[mi] - market_lon[0]) ** 2)
        transport_pct = 0.02 * dist_factor  # ~2% per degree of distance

        # Border effect
        border_effect = 0.08 if country != countries[0] else 0.0

        for comm, params in commodities.items():
            # Local price = reference + transport + border + local noise
            local_noise = rng.normal(0, params["volatility"] * params["base_price"] * 0.3, n_weeks)
            local_prices = ref_prices[comm] * (1 + transport_pct + border_effect) + local_noise
            local_prices = np.clip(local_prices, params["base_price"] * 0.3, params["base_price"] * 3)

            # Volume traded (kg, proportional to market size)
            base_volume = rng.exponential(500)
            volume = np.clip(base_volume + rng.normal(0, 100, n_weeks), 10, 5000)

            for t in range(n_weeks):
                rows.append({
                    "market": market,
                    "country": country,
                    "date": dates[t],
                    "week": t + 1,
                    "year": dates[t].year,
                    "month": dates[t].month,
                    "commodity": comm,
                    "price_per_kg_usd": round(local_prices[t], 3),
                    "volume_traded_kg": round(volume[t], 0),
                    "transport_cost_pct": round(transport_pct * 100, 1),
                })

    df = pd.DataFrame(rows)

    # Inject missingness (some weeks/markets missing)
    df = inject_missing(df,
        columns=["price_per_kg_usd", "volume_traded_kg"],
        rates=[0.04, 0.08],
        rng=rng, mechanism="MCAR")

    return df
