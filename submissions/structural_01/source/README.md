# House Edge structural forecaster

Exact Bayesian inference over the temporal rules recovered from the released data. For
each machine it weighs lag-pair, direction and sign-flip hypotheses against the history
before the cutoff, pools flip evidence across the episode, and infers which payout table
applies. A small fitted logit correction uses residuals from the same pre-cutoff history.

Run: `python predict.py --data /data --out /output/submission.csv`

Dependencies: Python 3.11+ and NumPy, pinned in requirements.txt. No network, API keys,
GPU or retraining. Inference is deterministic, so no random seed applies at prediction time.
`params.json` holds every saved parameter. `training-information.json` records the data
hashes, versions and validation scores.

Training, from the participant kit:
`python -m solution.fit_structural --data public --out runs/<new_folder>`
Training is deterministic and uses only organizer-released data. Condition labels select
stable episodes for pool discovery and are never read at inference. Validation true
probabilities fit the payout tables and logit correction only.
