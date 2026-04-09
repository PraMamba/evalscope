# Repository Guidelines

## Project Structure & Module Organization

- `evalscope/`: main Python package (CLI in `evalscope/cli`, core API in `evalscope/api`, benchmarks in `evalscope/benchmarks`, integrations in `evalscope/backend`, WebUI in `evalscope/app`).
- `tests/`: automated tests grouped by area (e.g., `tests/cli`, `tests/perf`, `tests/rag`, `tests/vlm`).
- `docs/`: Sphinx documentation sources (`docs/en`, `docs/zh`) plus helper scripts in `docs/scripts`.
- `examples/`: runnable examples and task recipes.
- `requirements/`: pinned requirement sets used by extras (see `pyproject.toml` optional-dependencies).

## Build, Test, and Development Commands

- Install (editable): `make install` (or `pip install -e .`).
- Dev setup: `make dev` (installs `.[dev,perf,docs]` + `pre-commit`).
- Lint/format: `make lint` (runs `pre-commit run --all-files`).
- Tests (CI style): `python -m pytest tests/cli/test_all.py::TestRun::test_ci_lite -v` and `python -m pytest tests/perf/test_perf.py -v`.
- Docs: `make docs` (all), `make docs-en`, `make docs-zh`.

## Coding Style & Naming Conventions

- Python 3.10+; 4-space indentation; max line length 120.
- Tools: `yapf` (format), `isort` (imports), `flake8` (lint); configs live in `setup.cfg` and `.pre-commit-config.yaml`.
- Prefer clear module names and keep public APIs under `evalscope/api` and CLI entrypoints under `evalscope/cli`.

## Testing Guidelines

- Test runner: `pytest` (installed via `.[dev]`), with many tests written in `unittest` style.
- Naming: `tests/**/test_*.py`, classes `Test*`, methods `test_*`.
- Some tests are gated by `TEST_LEVEL_LIST` (example: `TEST_LEVEL_LIST=0,1 python -m unittest discover tests`).

## Commit & Pull Request Guidelines

- Commits in history commonly use `feat: ...`, `fix: ...`, `feat(scope): ...` (e.g., `feat(benchmark): ...`), sometimes with tags like `[Fix]`, and often end with `(#1234)`.
- PRs: include a short description + rationale, link relevant issues, add/adjust tests, and include screenshots for `evalscope/app` UI changes. If you add optional deps, update `requirements/*.txt` and the corresponding extra in `pyproject.toml`.

## Security & Configuration Tips

- Keep API keys in environment variables or a local `.env` (CI uses `DASHSCOPE_API_KEY`); never commit secrets.
