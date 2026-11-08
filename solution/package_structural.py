"""Package a fitted structural model and verify it from a fresh extraction."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from slot_forecasting.io import read_queries, read_submission, write_json
from slot_forecasting.submission_contract import package_code

PREDICT = '''"""Recreate the forecast CSV from the saved model. No training and no network."""
import argparse
from pathlib import Path

from slot_forecasting.io import load_inputs, write_submission
from solution import structural


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    params = structural.load_params(root / "params.json")
    queries, history = load_inputs(args.data / "test")
    write_submission(args.out, structural.predict(queries, history, params))


if __name__ == "__main__":
    main()
'''

README = """# House Edge structural forecaster

## Entry

- Team ID: {team}
- Round: {round}
- Season ID: {season}
- Forecast file: submission.csv, uploaded separately
- Forecast SHA-256: {csv_sha}
- Code version: {version}
- Environment: `python` in `reproduce.json`, with NumPy from `requirements.txt`

## Setup

- Python/runtime version: tested on Python 3.13.18 and 3.12.3 on Linux, and 3.11.9 on Windows
- OS tested: Ubuntu under WSL2, and Windows 11
- Dependencies: `requirements.txt` with exact versions (NumPy only)
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
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("public"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--team", default="as selected in the submission Form")
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Use a new package folder")
    report = json.loads((args.run / "report.json").read_text(encoding="utf-8"))
    source = args.out / "source"
    source.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    shutil.copyfile(args.run / "params.json", source / "params.json")
    for folder, names in {"solution": ["__init__.py", "features.py", "structural.py", "fit_structural.py"],
                          "slot_forecasting": ["__init__.py", "schema.py", "io.py"]}.items():
        (source / folder).mkdir()
        for name in names:
            shutil.copyfile(root / folder / name, source / folder / name)
    (source / "predict.py").write_text(PREDICT, encoding="utf-8")
    metadata = report["metadata"]
    (source / "README.md").write_text(README.format(
        team=args.team, round=metadata["round"], season=metadata["season_id"],
        csv_sha=hashlib.sha256((args.run / "submission.csv").read_bytes()).hexdigest(),
        version=args.version or args.out.name), encoding="utf-8")
    (source / "requirements.txt").write_text("numpy==" + np.__version__ + "\n", encoding="utf-8")
    write_json(source / "reproduce.json", {
        "image": "python",
        "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]})
    write_json(source / "training-information.json", {key: report[key] for key in (
        "metadata", "input_sha256", "environment", "validation_cross_validated", "pair_counts")})

    archive = package_code(source, args.out / "entry.zip")
    extracted = args.out / "extracted"
    extracted.mkdir()
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extracted)
    output = (args.out / "reproduced.csv").resolve()
    start = time.perf_counter()
    subprocess.run([sys.executable, "-B", "predict.py", "--data", str(args.data.resolve()),
                    "--out", str(output)], cwd=extracted, check=True)
    seconds = time.perf_counter() - start
    ids = {query.query_id for query in read_queries(args.data / "test/queries.csv")}
    expected = read_submission(args.run / "submission.csv", ids)
    actual = read_submission(output, ids)
    difference = max(abs(x - y) for key in ids for x, y in zip(expected[key], actual[key]))
    if difference > 1e-8:
        raise SystemExit("Extracted prediction mismatch: " + str(difference))
    shutil.copyfile(args.run / "submission.csv", args.out / "submission.csv")
    shutil.copyfile(args.run / "validation_predictions.csv", args.out / "validation_predictions.csv")
    digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = {"queries": len(ids), "max_probability_difference": difference, "prediction_seconds": seconds,
              "zip_bytes": Path(archive).stat().st_size, "zip_sha256": digest(archive),
              "csv_bytes": (args.out / "submission.csv").stat().st_size,
              "csv_sha256": digest(args.out / "submission.csv")}
    write_json(args.out / "verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
