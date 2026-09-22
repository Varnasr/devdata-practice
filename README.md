# DevData Practice

[![Website](https://img.shields.io/badge/Docs-varnasr.github.io%2Fdevdata--practice-blue)](https://varnasr.github.io/devdata-practice/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![GitHub Last Commit](https://img.shields.io/github/last-commit/Varnasr/devdata-practice)](https://github.com/Varnasr/devdata-practice/commits/main)
[![Part of ImpactMojo](https://img.shields.io/badge/Part%20of-ImpactMojo-orange)](https://www.impactmojo.in)

**Realistic, large-scale practice datasets for development economics — 36 generators, 840,000+ rows.**

Built for researchers, students and practitioners who need data shaped like a
real survey: the same variables, the same awkward missingness, the same
structure, without waiting on a data request.

**Full documentation:** [varnasr.github.io/devdata-practice](https://varnasr.github.io/devdata-practice/)

---

## About

DevData Practice generates synthetic datasets built to the *shape* of development
sector surveys: the variables a DHS or LSMS instrument collects, laid out the way
it lays them out, with realistic missingness, partial compliance and attrition.

It does not reproduce any survey's distributions, and the figures it produces are
not estimates of anything. A stunting rate here is a number the code was told to
produce, not a measurement. If you need to check a pipeline against published
figures, use a real recode and a published table. `TRUTH.md` says exactly which
parameters are in the data and which estimand recovers each one.

The data is designed for:

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

Generated files are saved to `./output` as CSV files. Use `--output` to change
the directory and `--format parquet` to change the format.

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
numpy==2.4.6
pandas==3.0.6
scipy==1.17.1
pyarrow==25.0.1
```

Python 3.11 or higher. Versions are pinned exactly, not floored: see `CLAUDE.md`
for why. `requirements-dev.txt` adds pytest.

An earlier version of this section listed `faker>=15.0.0`. Nothing in the
repository imports it.

---

## Checking your answer

Every generator encodes parameters on purpose. `TRUTH.md` records what they are
and which estimand recovers each one, so an exercise can be marked rather than
guessed at.

The one worth reading before you use `rct_experiment`: its intention-to-treat
effect is **not** a fixed number. Take-up is drawn `U(0.65, 0.85)` for each arm on
each run, so the ITT moved between +0.089 and +0.203 log points across five seeds.
Only the complier effect is stable, and only when computed against the
pure-control villages (`spillover_risk == 0`). Controls inside a treatment village
receive a +3% spillover; using them as the comparison biases every ITT toward
zero by about 0.011 log points, which is the lesson the design exists to teach.

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q          # 233 tests, about two minutes
```

CI runs the suite on Python 3.11 and 3.12, then writes every dataset to CSV and
Parquet and checks the files are non-empty.

The largest part of the suite is a guard against degenerate columns, and it exists
because of a specific defect. `rng.binomial(1, 0.55)` without a size argument
returns a *scalar*, which numpy broadcasts across the whole column: nothing
raises, the row count is right, the file writes, and the variable is a constant.
Nineteen call sites across nine generators were affected. Among the fifteen
constant columns were four asset variables in `targeting`, a proxy-means-test
dataset whose asset predictors did not vary, and the three dropout barriers in
`girls_education`, all identically zero.

It was invisible because CI ran `python generate.py --list`, which imports no
generator at all. Thirty-six modules were unexecuted by any automated check.

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
