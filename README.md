# DevData Practice

Realistic, large-scale practice datasets for development economics. 10 generators covering household surveys, RCTs, health, education, agriculture, labor markets, microfinance, poverty targeting, panel data, and market prices.

**Documentation:** [https://varnasr.github.io/devdata-practice/](https://varnasr.github.io/devdata-practice/)

## Quick Start

```bash
pip install -r requirements.txt
python generate.py              # Generate all 10 datasets
python generate.py --list       # See available datasets
python generate.py rct_experiment labor_market  # Generate specific ones
```

## License

MIT — see [LICENSE](LICENSE).
