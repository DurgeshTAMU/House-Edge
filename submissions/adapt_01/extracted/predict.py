"""Recreate the forecast CSV from the saved model. No training and no network."""
import argparse
import platform
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_numpy():
    """Use an installed NumPy; otherwise unpack the matching bundled wheel into scratch space."""
    try:
        import numpy  # noqa: F401
        return "installed"
    except ImportError:
        pass
    tag = "cp%d%d" % sys.version_info[:2]
    machine = platform.machine().lower().replace("amd64", "x86_64").replace("arm64", "aarch64")
    for wheel in sorted((ROOT / "wheels").glob("numpy-*-%s-%s-*.whl" % (tag, tag))):
        if not wheel.name.endswith("_%s.whl" % machine):
            continue
        target = Path(tempfile.mkdtemp(prefix="house_edge_numpy_"))
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
        sys.path.insert(0, str(target))
        import numpy  # noqa: F401
        return "bundled " + wheel.name
    raise SystemExit("NumPy is not installed and no bundled wheel matches %s %s" % (tag, machine))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print("NumPy:", ensure_numpy(), file=sys.stderr)
    from slot_forecasting.io import load_inputs, write_submission
    from solution import structural
    root = ROOT
    params = structural.load_params(root / "params.json")
    queries, history = load_inputs(args.data / "test")
    write_submission(args.out, structural.predict(queries, history, params))


if __name__ == "__main__":
    main()
