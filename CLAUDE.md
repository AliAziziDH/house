# House Prices Kaggle & Decision Hub

## Stack
- Runtime: Python 3.12 (inside Dev Container slim-bookworm)
- Dependency Manager: Astral uv (volume-cached)
- Solvers: GLPK (system), COIN-OR CBC (system)
- Libraries: pandas 3.0.5, scikit-learn 1.9.0, pyomo 6.x

## Project Directory Layout
- `src/`          # Production-grade source code (separation of math vs UI)
- `tests/`        # Unit and mathematical integrity tests
- `data/`         # Kaggle datasets (train.csv, test.csv)

## Build, Test, & Execution Commands
- Run verification check: `uv run python src/verify_setup.py`
- Execute unit tests: `uv run pytest tests/`
- Lint code: `uv run ruff check src/`
- Auto-format code: `uv run ruff format src/`

## Coding Standards & Conventions
- Avoid raw mathematical calculations in LLM prompts; delegate strictly to Pyomo/SciPy.
- Write production-grade, object-oriented code. Map assets (Plants, Warehouses) as Python classes.
- Use explicit type annotations on all function signatures.
- Prioritize Gurobi-compatible or open-source compatible Pyomo formulations.