# DevData Practice

[![Website](https://img.shields.io/badge/Docs-varnasr.github.io%2Fdevdata--practice-blue)](https://varnasr.github.io/devdata-practice/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/Varnasr/devdata-practice)](https://github.com/Varnasr/devdata-practice/commits/main)
[![Part of ImpactMojo](https://img.shields.io/badge/Part%20of-ImpactMojo-orange)](https://www.impactmojo.in)

**Realistic, large-scale practice datasets for development economics — 36 generators, 840,000+ rows.**

Built for researchers, students, and practitioners who need real-feeling data modelled on DHS, NFHS, ASER, and other major development survey frameworks.

**Full documentation:** [varnasr.github.io/devdata-practice](https://varnasr.github.io/devdata-practice/)

---

## About

DevData Practice generates synthetic datasets that closely mirror the structure, variable distributions, and statistical properties of real development sector surveys. The data is designed for:

- **Learning** — practice data analysis, MEL, and econometrics without needing access to restricted datasets
- **Teaching** — ready-made datasets for classroom exercises, workshops, and tutorials
- **Prototyping** — build and test tools against realistic data before connecting to real sources
- **Demonstration** — showcase analysis workflows without sharing confidential programme data

All datasets are synthetic — no real individuals are represented.

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/Varnasr/devdata-practice.git
cd devdata-practice

# Install dependencies
pip install -r requirements.txt

# Generate all 36 datasets
python generate.py

# List available datasets
python generate.py --list

# Generate specific datasets
python generate.py rct_experiment labor_market household_survey
```

Generated files are saved to the `data/` directory as CSV files.

---

## Available Generators (36)

| Category | Generators |
|----------|-----------|
| **Health & Nutrition** | `health_nutrition`, `public_health`, `wash` |
| **Education** | `education`, `girls_education`, `irt_assessment` |
| **Livelihoods & Labour** | `livelihoods`, `labor_market`, `decent_work`, `microfinance` |
| **Gender & Social** | `gender_programme`, `care_economy`, `intersectionality`, `social_emotional_learning` |
| **Agriculture & Environment** | `agriculture`, `agri_value_chain`, `climate_resilience`, `environmental_justice` |
| **Governance & Policy** | `governance`, `social_protection`, `ngo_finance` |
| **Impact Evaluation** | `rct_experiment`, `cost_effectiveness`, `targeting`, `panel_data` |
| **Surveys & Field Work** | `household_survey`, `field_survey_quality` |
| **Behaviour & Communications** | `behaviour_change`, `media_development`, `bcc` |
| **Economics & Markets** | `trade_markets`, `digital_access`, `humanitarian` |
| **Development Architecture** | `aid_effectiveness`, `advocacy_rights`, `community_development` |

---

## Dataset Design

Each generator produces datasets modelled on real-world survey frameworks:

| Framework | Modelled in |
|-----------|------------|
| NFHS / DHS | `health_nutrition`, `household_survey`, `gender_programme` |
| ASER | `education`, `girls_education` |
| IHDS | `household_survey`, `livelihoods` |
| J-PAL RCT designs | `rct_experiment`, `targeting` |
| IRT (Rasch/2PL) | `irt_assessment` |

Variable names, distributions, and correlation structures are calibrated to approximate real survey data. Row counts are configurable — default is ~23,000 rows per dataset.

---

## Project Structure

```
devdata-practice/
├── generate.py             # Main entry point
├── requirements.txt        # Python dependencies
├── generators/             # One file per dataset type (36 generators)
│   ├── __init__.py
│   ├── household_survey.py
│   ├── rct_experiment.py
│   ├── health_nutrition.py
│   └── ... (33 more)
├── docs/                   # Documentation source (GitHub Pages)
├── LICENSE
└── README.md
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scipy>=1.9.0
faker>=15.0.0
```

Python 3.9 or higher.

---

## Part of the ImpactMojo Ecosystem

DevData Practice is a [ImpactMojo Professional](https://www.impactmojo.in) tier resource, also available as open-source for self-hosted use.

**Related repositories:**
- [ImpactMojo](https://github.com/Varnasr/ImpactMojo) — Main platform
- [deveconomics-toolkit](https://github.com/Varnasr/deveconomics-toolkit) — R and Python Shiny apps for development econometrics
- [InsightStack](https://github.com/Varnasr/InsightStack) — MEL tools and calculators

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use DevData Practice in research or teaching, please cite:

```
Sri Raman, V. (2025). DevData Practice: Synthetic datasets for development economics [Software].
GitHub. https://github.com/Varnasr/devdata-practice
```

Or use the [CITATION.cff](CITATION.cff) file.
