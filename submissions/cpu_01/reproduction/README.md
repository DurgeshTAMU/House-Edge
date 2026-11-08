# House Edge CPU submission

Install pinned requirements.txt before the offline run. Python 3.13 tested on Windows; Linux organizer runtime not tested.
Run: python predict.py --data /data --out /output/submission.csv
Uses a trusted included pickle checkpoint; no fitting, network or validation answers at inference.
Chosen model: features_augmented; seed 20261107; CPU thread limit 2.
Training from participant repository: python -m solution.train --data public --out runs/new_run
Training settings and versions: training-report.json. Upload forecast CSV separately.
