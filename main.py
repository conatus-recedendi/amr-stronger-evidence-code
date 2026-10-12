"""Run the EM pipeline from the repository root: uv run main.py [options]."""
from pathlib import Path
import runpy
import sys


def main():
    experiment_dir = Path(__file__).resolve().parent / "emergent_misalignment_experiments"
    # Preserve the original runner's sibling imports and subprocess entry point.
    sys.path.insert(0, str(experiment_dir))
    runpy.run_path(str(experiment_dir / "run_experiment.py"), run_name="__main__")


if __name__ == "__main__":
    main()
