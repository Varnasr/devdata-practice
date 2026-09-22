# Ground truth

A practice dataset you cannot check your answer against is a worked example with
the answer torn off. This file records the parameters the generators actually
use, which estimand recovers each one, and how close you should expect to get.
Everything here is asserted by `tests/`, so it cannot quietly drift away from the
code.

Two warnings before the numbers.

**Not every true parameter is a stable target.** In `rct_experiment` the take-up
rate is drawn `U(0.65, 0.85)` afresh for each arm on each run. The intention-to-treat
effect is therefore a different number for every seed. It ranged from +0.089 to
+0.203 log points across five seeds at n = 60,000. Only the complier effect is
stable. If you are checking your work against a fixed number, check the LATE.

**Recovering a parameter is not the same as the parameter being there.** Several
of these are recoverable only under the identifying assumption the exercise is
meant to teach. The spillover figure below is recoverable because the design has
pure-control villages; drop them and it is not identified at all.

## rct_experiment

Outcome is `log(endline_consumption_usd) - log(baseline_consumption_usd)`.

| Quantity | True value | How to recover it |
|---|---|---|
| Complier effect, cash_transfer | `log(1.15) + h` = **+0.1893** | Wald: ITT ÷ take-up, against pure controls |
| Complier effect, cash_plus_training | `log(1.22) + h` = **+0.2484** | same |
| Complier effect, training_only | `log(1.08) + h` = **+0.1265** | same |
| Spillover on untreated neighbours | `log(1.03)` = **+0.0296** | exposed controls minus pure controls |
| Common time trend | `log(1.03)` = +0.0296 | absorbed by any control group; not separately identified |

`h = log(1 + 0.04 × P(female) + 0.06 × P(below-median baseline)) = +0.0496` is the
average heterogeneity loading among compliers. It is part of the complier effect
because the heterogeneity terms multiply only for units with `actually_treated == 1`.
If you compare against `log(1.15) = +0.1398` alone you will appear to over-recover
by about five log points, and the gap is this term, not an error.

Observed across seeds 1, 42, 123, 777 and 2026 at n = 60,000: Wald means of
+0.1873, +0.2480 and +0.1248, all within 0.002 of theory. `tests/test_rct_truth.py`
asserts a tolerance of 0.01.

**The control group you choose changes the answer.** Villages are randomised into
treatment (70%) or pure control (30%), then individuals within treatment villages
are randomised across arms. Control units inside a treatment village receive the
+3% spillover. Using all controls as the comparison therefore biases every ITT
toward zero by roughly 0.011 log points. Using `spillover_risk == 0` controls only
is the clean contrast.

**`spillover_risk` used to be unusable.** Before this was fixed it equalled 1 for
every control and 0 for every treated unit, an exact alias of the treatment dummy,
because randomisation was stratified within district so every district always
contained treated units. Any regression including both dropped a collinear term.
`tests/test_no_degenerate_columns.py` now fails if any column collapses that way.

## Calibration targets

These are design targets for a synthetic teaching dataset, not estimates from a
named survey. They are asserted in `tests/test_calibration.py` so a future change
to a generator cannot silently move them.

| Generator | Quantity | Target |
|---|---|---|
| `public_health` | PHQ-9 ≥ 10 (moderate or worse) | 12–18% |
| `public_health` | PHQ-9 ≥ 20 (severe) | 0.8–2.5% |
| `public_health` | PHQ-9 in 0–4 (minimal) | 45–65% |
| `public_health` | PHQ-9 support | reaches both 0 and the upper 20s |
| `girls_education` | mean distance to school, rural | 3.5–4.5 km |
| `girls_education` | mean distance to school, urban | 1.2–1.8 km |
| `rct_experiment` | take-up in each treatment arm | 0.60–0.90 |
| `targeting` | asset ownership (radio, mobile, bicycle) | matches its stated rate ± 0.05 |

PHQ-9 is built the way the instrument is: nine items each scored 0–3 and summed.
The previous version mapped a single latent normal through `9 × logistic(...) × 3`,
which is bounded and has no item-level variance, so the floor sat at 2 and
`depression_severe` was identically zero in every draw.

## What is deliberately *not* true here

- No generator reproduces a real survey. Country names and GDP per capita in
  `generators/utils.py` are plausible orders of magnitude, not current figures,
  and should not be cited.
- Nothing here is a benchmark. If you need to check a pipeline against published
  numbers, use a real recode with a published table.
