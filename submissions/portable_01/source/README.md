# House Edge portable tree submission

Python 3.11+ standard library only. No pip install, model retraining, pickle loading, network or APIs.
Run from this folder: python predict.py --data /data --out /output/submission.csv
model.json stores the frozen numeric trees and feature column order. solution/features.py reproduces training features.
Exported from the features_augmented checkpoint trained with seed 20261107 on organizer training data.
Parity with the original model checked on every public validation/test query before packaging.
Training reproduction in participant workspace: python -m solution.train --data public --out runs/new_run
Export command in participant workspace: python -m solution.export --run runs/new_run --data public --out submissions/new_portable
Training code and dependencies belong to the participant workspace; this archive is inference-only.
Upload submission.csv separately. Offline organizer paths: read-only /submission and /data; writable /output.
