# House Edge implementation

The requested plan is the attached House Edge -- TAMU Datathon 2026 PDF. The other attachment is the participant guide. The local kit metadata and scorer use 60/40, superseding the guide's older 80/20 examples. Submission transport also differs: follow kit RULES.md and organizer announcements.

## First increment

- Preserved all organizer files and SHA256.json entries unchanged.
- Saved and packaged the pooled_recent fallback under runs/baseline_pooled_recent and submissions/baseline_entry.zip.
- Added solution/features.py: indexed histories, exact lag values and missingness, tick and observed-count windows, smoothed/decayed distributions, conditional histories, drift, and earlier peer outcomes from the same episode.
- Added solution/models.py: fixed-seed CPU histogram boosting and probability blending. No condition labels, true probability targets, IDs, signal, machine type, or absolute cutoff enter the history-only feature model. Excluding context is a conservative initial choice because historical availability is undocumented. Organizer baseline models retain their supplied behavior.
- Added solution/train.py: query-only and 12-history-samples-per-machine models; equal total training weight per machine (eight per episode); fixed transition candidates; blends with pooled_recent. Candidate settings are in source. No random row split or early-stopping split. The query-only model has 768 training rows; actual augmented count is recorded in report.json.
- Added solution/predict.py and solution/package.py: saved checkpoint inference, pinned dependencies, official ZIP packaging and fresh-extraction comparison.
- Added tests/test_solution.py and requirements-solution.txt.

## Reproduce

Run from this repository with Python 3.13 and requirements-solution.txt installed. Model fitting uses CPU with numerical-library thread limits of two. No GPU or remote runtime is assumed. Choose fresh output directories; existing run directories are preserved.

```powershell
python -m unittest discover -s tests -v
python -m solution.train --data public --out runs/cpu_comparison_repeat
python check_submission.py runs/cpu_comparison_repeat/validation_predictions.csv --split validation
python check_submission.py runs/cpu_comparison_repeat/submission.csv
python -m solution.package --run runs/cpu_comparison_repeat --data public --out submissions/cpu_repeat
```

Direct saved-checkpoint inference:

```powershell
python -m solution.predict --data public --model runs/cpu_comparison_01/model.pkl --out runs/prediction_repeat.csv
```

## Evaluation limitations

Official squared probability errors are averaged by episode and condition and normalized against the frozen reference. These are not classification accuracies or hidden-test scores. Tuning/audit episode manifests are saved in folds.json. Model selection uses tuning only; selected and fallback audit scores are reported afterward. Baseline full-validation results were already displayed by the supplied starter, so the audit is reserved from new-model selection, not entirely unseen. Do not use this audit for subsequent hyperparameter selection.

Checkpoint reload and ZIP extraction are tested on the current Windows/Python environment. A fresh Linux environment and organizer sandbox are not available here; pinned packages require organizer setup. Pickle checkpoints are locally generated trusted artifacts, not safe loaders for arbitrary downloaded files. The standard-library fallback avoids third-party model dependencies.

No files are uploaded. Keep main and adaptation artifacts separate. For adaptation, pass its public folder to --data and use new output paths. Further feature tuning, machine-specific interaction models, and broader experiments are follow-up work after reviewing this increment.

## Measured first-run results

Selected model: features_augmented. CPU comparison runtime: 106.85 seconds.

| Model | Full validation /60 | Audit /60 |
|---|---:|---:|
| Pooled-recent fallback | 30.825854 | 30.564508 |
| Historical-feature boosting | 32.107314 | 31.344264 |

Selected normalized loss ratio: 0.868733. Improvements remain preliminary; no test scores are available.
Training rows: {'features_augmented': 9984, 'features_queries': 768}. Selected checkpoint reload maximum probability difference: 0.0.
Eight unit tests passed. Both forecast files passed the official checker. Baseline fresh-folder reproduction difference was zero. See submissions/cpu_01/reproduction.json for selected ZIP reproduction results. Original kit hash verification passed.

Next increment: review per-episode errors and predeclare a new evaluation protocol before further tuning; test pinned dependencies in the organizer Linux environment.

## Second increment: portable inference and diagnostics

Recommended matching submission pair now lives in submissions/portable_01/submission.csv and submissions/portable_01/entry.zip. The frozen model and validation score are unchanged (32.107314/60); the inference format is now JSON trees and standard-library Python. No NumPy, scikit-learn, pip install, pickle, network, or retraining is needed for inference. Older submission artifacts remain preserved.

Added solution/portable.py (numeric tree traversal and stable softmax), solution/export.py (strict supported-model export and complete prediction parity checks), solution/diagnose.py (tuning-only error breakdown and training-fold manifest), and tests/test_portable.py (numeric/missing thresholds and softmax stability).

Ten tests passed. All validation and test probabilities match the original model within 2.22e-16. Windows reproduction with site packages disabled is exact. The extracted ZIP also ran in Ubuntu WSL2 Python 3.12.3 with site packages disabled, affinity restricted to two CPUs and a 16 GiB address-space limit. Measured Linux runtime was 9.09 seconds and peak RSS about 162 MiB. Linux probability difference: 1.1102230246251565e-16. This is a real Linux reproduction, not a test of the organizer's exact Python 3.13 container. The ZIP is 121,656 bytes. Both final CSV files pass the official checker. All original kit hashes still match.

Reproduce export and diagnostics (use fresh output paths):

```powershell
python -m unittest discover -s tests -v
python -m solution.export --run runs/cpu_comparison_01 --data public --out submissions/portable_repeat
python -m solution.diagnose --out runs/diagnostics_repeat
```

To predict without any third-party packages from an extracted portable ZIP:

```sh
python -B -S predict.py --data /absolute/path/to/public --out /absolute/path/to/submission.csv
```

Diagnostics are in runs/diagnostics_01/report.json and query_errors.csv. On 48 tuning episodes, the selected model improved 35 and lowered average error in all four conditions. The zero-payout probability accounts for most remaining squared error. The 20x component slightly regressed. The paired episode bootstrap interval is post-selection and must not be presented as independent confirmation.

runs/diagnostics_01/train_cv_protocol.json establishes three disjoint, condition-stratified folds containing only training episodes. Future bounded candidate comparisons should score held-out terminal observed outcomes with equal episode/condition weighting. This observed-outcome Brier loss is noisy and is not the official true-probability metric. Historical samples from a held-out episode must never enter that fold's fit. No new model was selected or tuned against the old audit in this increment.

Next: execute a small predeclared training-only cross-validation comparison focused on zero-payout probability and machine-specific temporal interactions. Do not install a new dependency or change the selected submission solely on a small tuning gain.

## Third increment: training-only capacity comparison

Ran the predeclared configs/train_cv_01.json experiment on three condition-stratified folds covering all 96 training episodes. All candidates used the same 9,984 examples and 301 history-only features. A held-out episode contributed no examples to fitting; only its terminal queries were scored. Each candidate produced 768 complete out-of-fold predictions.

| Candidate | Observed-outcome Brier | Zero-payout Brier |
|---|---:|---:|
| higher_capacity | 0.550235 | 0.211317 |
| incumbent | 0.544956 | 0.207401 |
| regularized | 0.545539 | 0.207850 |

Lower is better. These observed-outcome losses are not official true-probability losses or competition points. The incumbent won; neither alternative passed the fixed promotion rule (at least 0.5% lower total error and improvement on at least two folds). No new checkpoint was fitted on all data, and validation/test data were not read. Recommended submission remains submissions/portable_01/submission.csv plus its matching entry.zip, with the previously measured validation score 32.107314/60.

Experiment runtime: 278.73 seconds on local CPU with two numerical-library threads. Four new tests cover historical-row grouping, bad folds, promotion rules, and score weighting; all 14 tests passed. Out-of-fold CSVs validated against all 768 train IDs. Original organizer hashes and the recommended submission hashes remain unchanged.

Added solution/training_data.py, solution/cross_validate.py, configs/train_cv_01.json, and tests/test_cross_validate.py. Updated the exporter to record actual selected-model settings and training command in future packages. Existing ZIPs were not changed.

Reproduce this fixed experiment with a fresh output directory:

```powershell
python -m unittest discover -s tests -v
python -m solution.cross_validate --config configs/train_cv_01.json --data public --out runs/train_cv_repeat
```

Full results, runtimes, hashes, and decisions: runs/train_cv_01/report.json. Per-query predictions: oof_incumbent.csv, oof_regularized.csv, oof_higher_capacity.csv. The stored training_features.npz is a local cache, not a submission dependency.

Next research direction: richer machine-specific conditional temporal features rather than another capacity sweep. Keep any new candidate set small and declared in advance; repeated use of these CV folds can also overfit selection. The old validation audit remains unavailable for new tuning.

## Fourth increment: machine-local temporal candidate

Ran the predeclared `configs/temporal_cv_01.json` comparison on the same three
training-only, condition-stratified episode folds. The candidates used short and
long binary/ternary temporal contexts plus fixed blends with the incumbent; no
validation or test inputs were used during candidate selection. `binary_short_w1.0`
was selected with observed-outcome Brier `0.4651363421` versus the incumbent
`0.5449563491`, improving all three folds and exceeding the fixed 0.5% promotion
threshold.

The one allowed public-validation evaluation scored **39.324354 / 60**, compared
with **32.107314 / 60** for `portable_01`. Per-condition ratios were stable
0.259043, short_history 0.750994, drift 0.603113, and shared_change 0.489939.
This is a public-validation result and is not a hidden test score; no claim about
the final event ranking is made.

The recommended new pair is `submissions/temporal_02/submission.csv` and
`submissions/temporal_02/entry.zip`. The package uses standard-library inference,
contains the frozen temporal model and blend metadata, and reproduced the test
CSV from a fresh extraction with maximum probability difference
`8.881784197001252e-16`. The official checker accepted both the packaged CSV and
the reproduced CSV. The previous `submissions/portable_01` pair remains the
rollback option.

Validation completed with:

```powershell
python -m unittest discover -s tests -v
python check_submission.py submissions/temporal_02/validation_predictions.csv --split validation
python check_submission.py submissions/temporal_02/submission.csv
```

The unit suite reported 20 tests passed. No files were uploaded.


## Fifth increment: structural Bayesian model

Validation true probabilities exposed the simulator's rule family, and a generative model now
replaces frequency estimation. `solution/structural.py` holds inference and
`solution/fit_structural.py` learns every parameter from a round's public data.

**Recommended pair:** `submissions/structural_final/submission.csv` and
`submissions/structural_final/entry.zip`. Rollback: `submissions/temporal_02`.

| Model | Validation points /60 | R |
|---|---:|---:|
| pooled_recent fallback | 30.83 | |
| temporal_02 | 39.32 | 0.526 |
| Structural, fixed hand-set parameters | 56.75 | 0.057 |
| + logit correction (cross-validated) | 57.32 | 0.047 |
| + joint win/size inference (cross-validated) | 58.03 | 0.034 |
| + regime term in the win logit (cross-validated) | 58.86 | 0.019 |
| + per-type w and nu, final (cross-validated) | **58.91** | 0.018 |

Final condition ratios: stable 0.0108, short_history 0.0413, drift 0.0109, shared_change 0.0109.
Episode-bootstrap 95% interval: 58.60 to 59.18. These are public-validation scores, not hidden
test scores. The structure was found by inspecting validation targets, so 58.91 is post-selection
even though the correction is cross-validated by episode.

### What was discovered

- Win logit = base[type] + 0.34*signal + 1.02*(mean win across the episode at lag 3 - 0.45)
  + sign(t)*w[type]*e(t) + nu[type]*r(t). w is about 2.33 for every type; nu is -0.26 steady,
  -0.43 balanced, -0.51 jackpot.
- e(t) is the parity of a machine's win flags at one lag pair from {(1,3), (2,5), (4,7)}.
- Two exact payout tables per type. Table 1 is active when (payout[t-L] >= 2) != d. L follows
  from the pair: 1, 2 and 4 respectively. d is machine-specific.
- shared_change flips sign(t) at one tick for the whole episode. drift flips each machine around
  an episode centre (spread about 12.6 ticks). stable and short_history never flip; the short
  evidence favours no flip by about 135 nats.
- Signal is valid at historical ticks: its effect doesn't vary over the history.
- A residual probe found the regime term. After it, a boosted residual model on extra features
  did worse out of sample than the linear correction (0.3865 vs 0.3754), so modelling stopped.
- Global parameters sit at their optimum: scaling w, gamma or kappa does not improve the score.

### Independent checks

| Check | Result |
|---|---|
| Unused train labels, observed Brier | 0.3429 vs temporal_02 out-of-fold 0.4651; better in all four conditions |
| Calibration, observed vs model-expected Brier | train 0.3429 vs 0.3689 (1.6 SE); validation 0.3813 vs 0.3846 |
| Inferred conditions on test, no labels | 48/48/48/48, matching metadata |
| Synthetic kit with different rules | all rules recovered; 39.87/40 (engine before the regime term) |
| Fresh ZIP extraction, Windows | maximum difference 0.0 |
| Fresh ZIP extraction, Ubuntu WSL2, two CPUs, read-only code | 1.63e-13, 14 seconds |
| Organizer file hashes | 37 of 37 match |
| Unit tests | 32 passed |

The Linux run used Python 3.12.3 with NumPy 2.3.4. The organizer image is Python 3.13, which was
not available here. The entry needs NumPy, pinned in its `requirements.txt`, unlike the earlier
standard-library entries. RULES.md says organizers prepare pinned packages.

### Reproduce

```powershell
python -m unittest discover -s tests
python -m solution.fit_structural --data public --out runs/structural_repeat
python check_submission.py runs/structural_repeat/validation_cv.csv --split validation
python -m solution.package_structural --run runs/structural_repeat --data public --out submissions/structural_repeat
```

Fitting is deterministic and takes about 2.5 minutes. `validation_cv.csv` holds the cross-validated
forecasts; `validation_predictions.csv` uses the correction fitted on all validation episodes and
scores slightly higher in-sample (58.96). Round two steps are in `ROUND_TWO_RUNBOOK.md`.


## Requirements audit

**Recommended pair is now `submissions/structural_final_v2/`.** Its CSV is byte-identical to
`submissions/structural_final/`. The ZIP changed only to follow `SUBMISSION_README.md` and to include
`solution/fit_structural.py`, which the README cites.

| Check | Result |
|---|---|
| Kit tuning/audit split: rules from train only, tables and correction from tuning only | audit 58.75, tuning 59.06 |
| Everything fitted on validation only, scored on unused train labels | same rules; Brier 0.3427 vs 0.4651 |
| Backtest on test histories, cutoffs moved back 1-6 ticks, 9,216 real outcomes | Brier 0.3440 vs 0.3441 expected; calibrated in every bin |
| Python 3.13.18, Linux, no network, two CPUs, 16 GiB limit | 10-12 seconds; 1.63e-13 |
| From-scratch refit | parameters identical; maximum difference 0.0 |
| Unit tests | 34 passed |
| Data integrity across splits | 8 machines per episode, one cutoff, no ticks at or after cutoff, no shared episodes |

Mean P(win) is lower on test (0.382) than validation (0.439). That is a real difference in the state at
the cutoff, not bias: the backtest above matches on test's own outcomes.

Limits: the audit half was already seen when the rules were discovered, so it is not pristine. The
correction added 0.04 points on audit against 0.55 on tuning, so treat about 58.75 as the conservative
estimate. Organizer `schema.py` and `io.py` are copied into the ZIP, as `build_entry.py --starter` does.
The team field reads "as selected in the submission Form"; set it with `--team`.


## Round-two stress test

`tests/stress_kit.py` builds simulated kits whose rules break main-round assumptions, with hidden state and a real
pooled reference fitted on each kit's training data. Scores are out of 40; the pooled reference scores 20.

| Scenario | Before generalising | After |
|---|---:|---:|
| Main-like baseline (control) | 39.50 | 39.50 |
| Same rules as main (control) | 39.32 | 39.32 |
| Single-lag rule | 21.98 | 39.42 |
| Three-lag rule | 20.36 | 39.30 |
| AND rule | 24.24 | 39.21 |
| Mixed XOR/AND/OR rules | 27.10 | 38.97 |
| Two flips per machine | 19.90 | 38.02 |
| Flip to partial strength | 20.81 | 34.43 |
| Payout table also flips | 37.20 | 37.20 |
| Weak rule | 36.94 | 36.87 |

Unchanged from the first run (38.6-39.5): new lags, five pairs, pools by machine type, size threshold 5, untied size
lags, early/late/ramp flips, short episodes also flipping, two shared lags, new payout tables.

Changes in `solution/structural.py` and `solution/fit_structural.py`:

- Rules are lag sets of 1-3 lags with truth tables learned from stable histories (BIC-penalised), not fixed XOR pairs.
- A fourth condition, a per-machine sticky regime over rule strength (7 levels), competes on evidence. It covers
  repeated and partial flips and gets essentially zero weight on real main data.
- When that regime dominates, parameter fitting follows its Viterbi path.
- `report.json` has `family_check`: PASS or WARN with reasons, plus informational notes.
- Inference was vectorised (about 6.6x faster, identical output to 5.6e-17).

On real main data the final code rediscovers exactly the three XOR pairs and scores 58.913/60 cross-validated
(58.912 before), family check PASS, fit time 620 seconds on an idle machine. 34 unit tests pass.

The main entry is unchanged: `submissions/structural_final_v2`. The generalised code is for round two only.
