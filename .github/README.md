# House Edge: structural model (TAMU Datathon 2026)

Our entry for the House Edge slot-payout forecasting challenge. The main-round model scores **58.91/60
cross-validated** on public validation; the pooled starter scores 30. The organizer's participant guide is
the root `README.md`.

## Setup

1. Clone this repo.
2. Copy the organizer kit's `public/` folder into the repo root. Organizer data is not in this repo.
3. Use Python 3.11+ with NumPy: `pip install -r requirements-solution.txt`.

## Check it works

```powershell
python -m unittest discover -s tests -p "test_*.py"     # 34 tests, about 5 minutes
python check_submission.py submissions/structural_final_v2/submission.csv
```

## Where things are

| Path | What it is |
| --- | --- |
| `submissions/structural_final_v4/` | **Main-round entry to upload (team TwoStack)**: `submission.csv` (SHA-256 `79c94fd0…`) and `entry.zip` (`ea1be2ce…`). Same model and CSV as v2, plus offline NumPy wheels so it also runs on a standard-library-only Python 3.13 runner |
| `submissions/structural_final_v3/` | Same as v4 without the team ID in its README |
| `submissions/structural_final_v2/` | Previous packaging of the same entry (relies on organizers installing `requirements.txt`) |
| `vendor/numpy-wheels/` | Offline Linux NumPy 2.3.4 wheels bundled into entries with `package_structural --wheels` |
| `solution/structural.py` | The model: exact Bayesian inference over the simulator's hidden rules and change points |
| `solution/fit_structural.py` | Fits everything from a kit's `public/` data: `python -m solution.fit_structural --data public --out runs/<name>` |
| `solution/package_structural.py` | Builds and self-checks the CSV and code ZIP for a fitted run |
| `ROUND_TWO_RUNBOOK.md` | **Step-by-step commands for the adaptation round**, plus how to read the `family_check` |
| `SOLUTION.md` | How the model was found, scores at each step, requirements audit, round-two stress test |
| `tests/` | Leakage, missingness, output-format and inference tests; `stress_kit.py` simulates round-two rule changes |
| `submissions/temporal_02/` | Earlier fallback entry (39.32/60) |

`runs/` (experiment outputs) is not in the repo. A fit regenerates it in about 10 minutes.

## Round two

Follow `ROUND_TWO_RUNBOOK.md` in a **separate copy** of the new adaptation kit. Copy in `solution/` and `tests/`,
fit, read `family_check` in `report.json`, check, then package with `--team`. Never overwrite the main entry.
