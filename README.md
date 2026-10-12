# Code for "Anthropomorphized Misalignment Research Needs Stronger Evidence"

This repository contains code accompanying the ICML 2026 position paper
"Anthropomorphized Misalignment Research Needs Stronger Evidence".

## Repository Structure

- `deception_experiments/`: code for deception/probe experiments, including probe stress tests, activation extraction, probe training/evaluation, MASK-related evaluation, dashboard utilities, and workflow documentation.
- `emergent_misalignment_experiments/`: scripts used for emergent-misalignment fine-tuning, free-form inference, and GPT-based evaluation.

## Quick start (uv project)

The root uv project runs the **8B emergent misalignment experiment**. Python 3.11
is selected by `.python-version`; dependencies and their resolved versions live in
`pyproject.toml` and `uv.lock`. Run these commands from the repository root:

```bash
uv sync --locked
uv run main.py --dry-run
uv run main.py --smoke-test --skip-judge --output ./outputs/em-smoke
uv run main.py --output ./outputs/em-aesthetic
```

The full run uses two CUDA GPUs for LoRA training, then generates and evaluates
base-model and fine-tuned responses. Set `HF_TOKEN` (with Llama model access) and
`OPENAI_API_KEY`, or copy `emergent_misalignment_experiments/.env.template` to
`.env` in that directory and fill it in. GPT evaluation makes up to 4,800 paid
requests before retries; add `--skip-judge` to run without API evaluation.
macOS can run dry-run and logic tests; actual experiments require CUDA.

`uv run main.py --help` lists options. The longer entry point,
`uv run emergent_misalignment_experiments/run_experiment.py`, also works.
`uv run` automatically syncs the environment, so a separate `uv sync` is optional.
Keep `uv.lock` in version control; `.venv/` is already ignored.

```bash
uv run python -m unittest discover -s emergent_misalignment_experiments/tests -v
```

The original EM scripts also use the root environment. For their optional 4-bit
mode on Linux, install `uv sync --extra quantization` and run with
`uv run --extra quantization ...`.

`deception_experiments/` remains an independent uv project because it pins a
different PyTorch version. Run its commands from that directory (or use
`uv run --project deception_experiments ...`); it is not part of the root lockfile.

## Reproducing Experiments

The most complete workflow documentation is in:

- `deception_experiments/README.md`
- `deception_experiments/experiments.md`
- `emergent_misalignment_experiments/README.md`

The full sweeps require model access, API keys, and substantial compute. The code uses `uv` for Python environment management.

## Environment Variables

Copy the relevant `.env.template` file and fill in local paths/API keys where needed. Do not commit `.env` files.

## License

This code is released under the MIT License. See `LICENSE`.
