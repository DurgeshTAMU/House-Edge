# House Edge structural forecaster

## Entry

- Team ID: TwoStack
- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Forecast file: submission.csv, uploaded separately
- Forecast SHA-256: 79c94fd08eff17e080550d775e7a4e9a68176bd4209f36be0bdc1b5676514eb1
- Code version: structural_final_v4
- Environment: `python` in `reproduce.json` (Python 3.13 standard library), plus NumPy as below

## Setup

- Python/runtime version: tested on Python 3.13.18 and 3.12.3 on Linux, and 3.11.9 on Windows
- OS tested: Ubuntu under WSL2, and Windows 11
- Dependencies: NumPy only, pinned in `requirements.txt` (numpy==2.3.4). No setup step or network is needed:
  if NumPy is not installed, `predict.py` unpacks the matching offline Linux wheel from `wheels/`
  into a temporary folder under `/tmp` (about 55 MiB) and imports it from there.
  Bundled wheels: `numpy-2.3.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, `numpy-2.3.4-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`, `numpy-2.3.4-cp313-cp313-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl`.
- Required hardware: CPU only; two cores and under 16 GiB RAM. No GPU.
- Put the released data at: `public/`, mounted at `/data` by the organizer rerun

## Recreate the forecast

Run from: the root of this archive (`/submission`)

```sh
python predict.py --data /data --out /output/submission.csv
```

- Output file: `/output/submission.csv`
- Required model/config files: `params.json`
- Random seeds and settings: inference uses no randomness. Training uses no random draws
  except a fixed cross-validation shuffle, seed 20261107, which only affects reported scores.
  Every setting is in `params.json`.
- What the command does: loads the saved model and predicts. It does not train.
- Measured runtime and hardware: about 15 seconds on two CPU cores under Linux
- Training command, if separate: from the participant kit, with `solution/fit_structural.py`
  in place: `python -m solution.fit_structural --data public --out runs/<new_folder>`
- Main-round data/model reused in adaptation, if any: none. Each round refits from its own data.

## Method

Exact Bayesian inference over the temporal rules recovered from the released data. For each
machine it weighs lag-pair, direction and sign-flip hypotheses against the history before the
cutoff, pools flip evidence across the episode, and infers which payout table applies. A small
fitted logit correction uses residuals from the same pre-cutoff history.

Training uses only organizer-released data. Condition labels select stable episodes for rule
discovery and are never read at inference. Validation true probabilities fit the payout tables
and logit correction only. `training-information.json` records data hashes, versions and scores.

## Local check

```sh
python check_submission.py submission.csv
```

- Checker result: valid
- Fresh-folder reproduction checked: yes. The largest probability difference was within 1e-12.
