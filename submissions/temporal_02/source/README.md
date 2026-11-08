# House Edge temporal submission

Python 3.11+ standard library only. No pip installation, network, pickle, API, or external file dependencies.
Run: python predict.py --data /data --out /output/submission.csv
Chosen model: binary_short_w1.0. Training/configuration metadata: training-information.json.
The temporal predictor updates machine-local counts using only supplied observations before cutoff.
temporal.json stores global prior and settings; blend.json fixes the mixture weight. model.json is included only if the boosted component is used.
Training from participant workspace: python -m solution.temporal_cv --config configs/temporal_cv_01.json --data public --out runs/new_run
The incumbent model/OOF references in that configuration are required for training, not for this self-contained inference archive.
Keep /submission and /data read-only; output is written to the requested /output path.
Upload the matching forecast CSV separately.
