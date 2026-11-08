"""Recreate the forecast CSV from the saved model. No training and no network."""
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
