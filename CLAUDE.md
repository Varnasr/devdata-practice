# devdata-practice

Synthetic datasets for practising development economics analysis. Thirty-six
generators, pure Python, no service dependencies, nothing fetched at runtime.

## Commands

```bash
pip install -r requirements-dev.txt
python generate.py --list                  # registry
python generate.py rct_experiment          # one dataset to ./output
python generate.py --rows 500              # all of them, small
python -m pytest tests/ -q                 # ~2 min, mostly the degeneracy guard
```

## Layout

`generate.py` holds `GENERATORS`, the registry mapping a dataset name to its
module, its size keyword and a description. `generators/` holds one module per
dataset, each exporting `generate(**kwargs) -> DataFrame`. `generators/utils.py`
holds the shared geography, missingness and id helpers. `TRUTH.md` is the answer
key. `tests/` is the guard.

A module on disk that is not in `GENERATORS` is invisible to every user; one in
`GENERATORS` that is not on disk crashes. `tests/test_generators_run.py` compares
the two, so neither can happen quietly.

## The bug this repository is shaped around

`rng.binomial(1, 0.55)` **without a size argument returns a scalar**, which numpy
broadcasts across the whole column. Nothing raises, the row count is right, the
file writes, and the variable is a constant. It was present at nineteen call
sites across nine generators and produced fifteen constant columns, including
four asset variables in `targeting`, a proxy-means-test dataset whose asset
predictors did not vary.

It appears in two shapes:

```python
owns_radio = rng.binomial(1, 0.55)                          # every row: the same draw
months_displaced = np.where(displaced, rng.exponential(14), 0)   # every displaced row: the same
```

The second shape is the dangerous one, because the column then holds *two*
values rather than one and so survives any "is this column constant" check. It
is why `tests/test_no_degenerate_columns.py` tests for perfect binary aliasing
as well as for constants.

The discriminator when reading code: **is the probability a scalar or an array?**
`barrier_cost` in `girls_education` was always fine because its probability is
`0.35 - 0.10 * receives_scholarship`, an array, so numpy returned a vector. Its
neighbours on the surrounding lines took scalar probabilities and were broken.

A scalar draw is correct when it is a *parameter* rather than a column: the
per-arm compliance rate in `rct_experiment`, the per-country intercept in
`panel_data`, anything inside a per-row loop. Do not add `size=n` to those. The
test to apply is whether the result becomes a column.

## Watch out for

- **Do not loosen a tolerance in `tests/` to make a failure go away.** The
  numbers in `TRUTH.md` are what a learner checks their answer against. If a test
  fails, either the generator changed or `TRUTH.md` is now lying.
- **The ITT in `rct_experiment` is not a fixed number.** Take-up is drawn
  `U(0.65, 0.85)` per arm per run, so the ITT moved between +0.089 and +0.203
  across five seeds. Only the complier effect (Wald: ITT ÷ take-up) is a stable
  target, and only against pure-control villages.
- **`spillover_risk` has a village level under it.** Villages are randomised
  into treatment (70%) or pure control (30%); the flag marks control units inside
  a treatment village. Before that existed, randomisation was stratified within
  district, every district therefore contained treated units, and the flag was an
  exact alias of the control dummy. If you flatten the design back to individual
  randomisation, the flag becomes useless again.
- **PHQ-9 is a sum of nine 0–3 items, not a transformed latent.** The previous
  `9 * logistic(...) * 3` construction was bounded and had no item-level variance:
  the floor sat at 2 and `depression_severe` was identically zero in every draw.
  If you touch the item severities, re-check the bands in `TRUTH.md`.
- **Requirements are pinned on purpose.** `pandas>=2.0` would pick up pandas 3,
  where text columns carry a `str` dtype rather than `object`, so any check keyed
  on `dtype == object` skips every text column without saying so. Raise the pins
  deliberately and together, then re-run the suite.
- **Nothing here is real data.** Country names and GDP per capita in
  `generators/utils.py` are plausible orders of magnitude, not current figures.
  Do not cite them, and do not describe any generator as reproducing a survey.

## Adding a generator

1. Write `generators/<name>.py` exporting `generate(n_x=..., seed=...) -> DataFrame`.
2. Register it in `GENERATORS` in `generate.py` with its size keyword and a
   one-line description.
3. Run `python -m pytest tests/ -q`. The degeneracy guard runs against every
   registered generator automatically, so you do not add a test for it.
4. If the generator encodes a parameter someone is meant to recover, put it in
   `TRUTH.md` and assert it in `tests/`.

## Testing

`.github/workflows/ci.yml` runs the suite on 3.11 and 3.12, plus an end-to-end
job that writes every dataset to CSV and Parquet and checks the files are
non-empty. Serialisation is a separate failure surface from generation: parquet
is stricter than csv about mixed types, so a frame that builds happily can still
fail to write.

CI used to run `python generate.py --list` and nothing else. That imports no
generator, so all thirty-six modules were unexecuted by any automated check.
