# Round two runbook

Round two (adaptation) is worth 40 points. New data: Saturday 9 PM Central. Deadline: Sunday 11 AM Central.
Main entries can't be replaced after 9 PM Saturday, so submit main first.

## Run this first

Extract the new kit into its own folder. Never overwrite this folder's `public/` or `submissions/`.

```powershell
# from the NEW kit folder, after copying in solution/ and tests/ from this folder
python -c "import json; m=json.load(open('public/metadata.json')); print(m['round'], m['max_points'])"
python -m unittest tests.test_structural
python -m solution.fit_structural --data public --out runs/adapt_01
python check_submission.py runs/adapt_01/validation_cv.csv --split validation
python check_submission.py runs/adapt_01/submission.csv
python -m solution.package_structural --run runs/adapt_01 --data public --out submissions/adapt_01 --team "TwoStack" --wheels vendor/numpy-wheels/numpy-2.3.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl vendor/numpy-wheels/numpy-2.3.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl vendor/numpy-wheels/numpy-2.3.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl
```

Copy `vendor/` into the new kit too. The participant guide says the `python` runner has only the standard library, so
`--wheels` bundles offline NumPy wheels that `predict.py` unpacks into `/tmp` when NumPy isn't installed (about 47 MB ZIP).

The first line must print `adaptation 40`. Fitting takes about 3 minutes on a laptop CPU.
Upload `submissions/adapt_01/submission.csv` and `submissions/adapt_01/entry.zip` under the adaptation Form.

`validation_cv.csv` is the honest score: each episode's correction was fitted without that episode.
Don't quote `validation_predictions.csv`, which is in-sample for the correction.

## What refits automatically

| Learned from the new data | Report field |
| --- | --- |
| Rule lag sets (1, 2 or 3 lags) and each one's learned truth table (XOR, AND, OR, ...) | `pair_counts`, `params.pairs`, `params.rule_tables` |
| Size threshold, size lags, pair-to-lag link | `size_refit`, `params.size_*` |
| Shared lag | `shared_lag_gain`, `params.kappa_lag` |
| Base logits, parity strength, regime effect, signal, shared effect | `params.base`, `w`, `nu`, `gamma`, `kappa` |
| Flip window and drift spread | `flips`, `params.tau_*`, `drift_sd` |
| Whether short episodes flip | `short_flips` |
| Payout tables | `table_spread`, `params.tables` |

## Is the model family still right?

Read `family_check.status` in `runs/adapt_01/report.json` first.

- `PASS` on main: loss ratio 0.018, condition agreement 1.0, catch-all regime share 0.0.
- `WARN` lists its reasons. A high `regime_share` with low `condition_agreement` means flips no longer follow
  one clean reversal; the catch-all regime model is then carrying those episodes.
- `notes` are informational. A weak but correctly learned rule lowers the score ceiling; it is not a break.

How the pipeline handled simulated rule changes (score /40; the pooled reference scores 20):

| Simulated change | Score | Check |
| --- | ---: | --- |
| Main-like rules, new lags, 5 pairs, new tables, early/late/ramp flips, 2 shared lags | 38.6-39.5 | PASS |
| Single-lag, three-lag, AND, or mixed XOR/AND/OR rules | 38.97-39.42 | PASS |
| Two flips per machine | 38.02 | WARN (drift ratio 0.12) |
| Payout table also flips | 37.20 | WARN |
| Weak rule (w 1.2) | 36.87 | WARN (short_history 0.20) |
| Flip to partial strength | 34.43 | WARN |

Fitting takes several minutes and up to about 30 minutes if the machine is busy. Don't run other heavy jobs alongside it.

Then also check:

1. `table_spread` is near zero (main: about 1e-15). A large value means more than two tables per type.
2. `pair_counts` has a few dominant pairs and little else. A flat spread means wins no longer follow one parity pair.
3. `validation_no_correction` and `validation_cross_validated` are in the mid-30s or higher out of 40.
   The pooled starter scores about 20, and the equivalent main result was 39.27/40 at the same R.
4. The `short_history` loss ratio is the worst but under about 0.1, and the other three are about 0.01–0.03.
5. The condition posterior still separates episodes. `discover_flips` reporting `kept_defaults` means flips changed.

## If a check fails

Repeat the residual probe that found each term in main, rather than tuning:

1. Fit the run, then compare `logit(true p_win)` with the model's logit on validation.
2. Correlate that residual with lags, regime, episode means and machine type.
3. A strong correlation is a missing term. Add it to `infer`, and to `fit_logit` so it's learned.
4. If the rule family has truly changed and can't be repaired in time, run the temporal fallback and submit whichever
   scores higher on validation:

```powershell
python -m solution.temporal_cv --config configs/temporal_cv_01.json --data public --out runs/adapt_temporal
```

That config points at main-round incumbent files, so it needs those paths updated first.

## Before uploading

- Upload both files before cutoff; the Form timestamp decides. The latest response counts even if invalid.
- Main and adaptation query IDs differ, so the main CSV can't be reused.
- Keep main artifacts untouched.
