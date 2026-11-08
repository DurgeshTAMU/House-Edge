# Round two (adaptation) and overfitting checks

Team TwoStack. Main and adaptation entries are both fitted by the same pipeline (`solution/fit_structural.py`),
which was designed on main data only before the adaptation kit was released.

## Entries

| Round | Folder | `submission.csv` SHA-256 | `entry.zip` SHA-256 | Validation score |
| --- | --- | --- | --- | --- |
| Main (80 pts per the adaptation kit; the main kit said 60) | `submissions/structural_final_v5/` | `79c94fd0…` | `a288aaa8…` | 58.91 / 60 cross-validated |
| Adaptation (20 pts) | `submissions/adapt_01/` | `094559b1…` | `d1ff1a7d…` | 19.70 / 20 cross-validated |

The pooled reference scores half of each round's maximum (30/60 and 10/20). The adaptation kit's `RULES.md`
gives a 80/20 split. Its scoring code (`slot_forecasting/rounds.py`) and `metadata.json` agree with that split.

## What changed in round two

The refit rediscovered the rules from the adaptation data alone:

| | Main | Adaptation |
| --- | --- | --- |
| Win-rule lag pairs (all XOR) | (1,3), (2,5), (4,7) | (1,4), (2,6), (3,8) |
| Payout-size lags | 1, 2, 4 | 1, 2, 3 |
| Shared-machine lag | 3 | 3 |

The main model run unchanged on adaptation validation scored 7.73/20, below the reference, so refitting was
necessary. Fit time 11 minutes; `family_check` PASS (condition agreement 1.0, catch-all regime share 0.0).

## Is either model overfitted?

Every check scores the model on data that the fit did not use, through the real fit pipelines.

| Check | Main (submitted v5 code and model) | Adaptation (`adapt_01`) |
| --- | --- | --- |
| Cross-validated score (correction fitted without each episode) | 58.91 / 60 | 19.70 / 20 |
| Fit on validation half A only, scored once on unseen half B | 58.73 | 19.71 |
| Fit on half B only, scored once on unseen half A | 58.84 | 19.65 |
| Rules found independently by each half-fit | same three pairs both times | same three pairs both times |
| 95% interval, episode bootstrap of the cross-validated score | 58.59 – 59.17 | 19.61 – 19.77 |
| Backtest on **test** histories (never used by any fit), observed vs model-expected Brier, 9,216 rolled-back ticks | 0.3440 vs 0.3441 (gap −0.0002 ± 0.0038) | 0.3489 vs 0.3523 (gap −0.0034 ± 0.0039) |
| Same backtest on validation histories | gap +0.0032 ± 0.0059 | gap −0.0020 ± 0.0051 |

Halves use the kit's own stratified `episode_folds` split. Train and test are unchanged.

Fitted constants came out almost identical in the two rounds, although the rule lags changed:

| Constant | Main | Adaptation |
| --- | --- | --- |
| Rule strength `w` (steady / balanced / jackpot) | 2.33 / 2.32 / 2.37 | 2.32 / 2.33 / 2.31 |
| Payout-table effect `nu` | −0.27 / −0.43 / −0.51 | −0.24 / −0.42 / −0.50 |
| Signal `gamma` | 0.34 | 0.34 |
| Shared-machine `kappa` | 1.02 | 1.05 |

**What the checks show.**

- **Held-out scores:** held-out scores match the cross-validated ones within their bootstrap intervals.
- **No validation advantage:** the models do no better on validation histories than on test histories, and the fitted constants repeat across rounds.
- **Remaining error:** the adaptation residual probe (`tools/verification/residual_probe.py`) found the remaining error unlearnable. An out-of-fold ridge on 89 lag, lag-pair and type features explained 2.2% of the residual win logit, and applying it made the error worse (0.00297 vs 0.00285).
- **Where the error sits:** 87% of what remains is in win probability, concentrated in short-history episodes and jackpot machines, where the hidden state is least identifiable.

No tuning was done on the adaptation data.

## Round-two submission checklist (adaptation kit `README.md` section 7)

| Requirement | `submissions/adapt_01` |
| --- | --- |
| CSV: every adaptation test query once, six columns, rows sum to 1, UTF-8, at most 10 MB | `check_submission.py`: VALID, 1,536 queries, adaptation, test; 0.21 MB |
| ZIP at most 100 MB, `README.md` and `reproduce.json` at root, no CSV inside | 30.5 MB; the organizer's `build_entry.py --source` rebuilds the identical file list and contents |
| `reproduce.json` exactly `python predict.py --data /data --out /output/submission.csv`, image `python` | Yes |
| Extra libraries pinned in `requirements.txt` | `numpy==2.3.4`. Offline wheels for Python 3.13 x86-64 and ARM are also bundled, so it also runs if NumPy is not installed. |
| No network, no files outside the ZIP and `/data` | Loads the saved `params.json`; does not train |
| Recreated probabilities within 1e-8, under 30 min and 16 GiB | Windows fresh extraction: 0.0 difference. Linux Python 3.13 with no NumPy, no network, read-only code: 1.6e-13, 18 s |
| README with team, round, dependency versions, seeds, data path, command | Team TwoStack, round adaptation, season `5fffdc4d0c700bbf61c3f911` |

## Reproduce the checks

From the adaptation kit with `solution/`, `tests/` and `vendor/` copied in:

```sh
python -m solution.fit_structural --data public --out runs/adapt_01
python check_submission.py runs/adapt_01/validation_cv.csv --split validation
PYTHONPATH=. python tools/verification/make_half_kits.py public halves        # then fit halves/A/public and halves/B/public
PYTHONPATH=. python tools/verification/score_holdout.py public <run_A> halves/split.json B_audit 20
PYTHONPATH=. python tools/verification/backtest2.py public runs/adapt_01/params.json
python tools/verification/residual_probe.py public runs/adapt_01/validation_cv.csv
```
