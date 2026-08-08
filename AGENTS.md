# Agent Guidelines & Repository Conventions

## Environment & Tooling
- This repository uses `uv` for dependency and environment management.
- Execute Python commands and scripts using `uv run`.

## Python Pathing Rules
- Always prepend `PYTHONPATH=.` when running Python scripts or tests from the root directory to ensure imports from `src/` resolve properly:
  `PYTHONPATH=. uv run python <script_path>`

## Testing & Validation Commands
- **Run Unit Tests**:
  ```bash
  PYTHONPATH=. uv run pytest tests/ -v
  ```
- **Run Pipeline Diagnostics**:
  ```bash
  uv run python src/diagnose_pipeline.py
  ```

## Verification Rule
- **Mandatory Pre-Commit Verification**: Always run tests and/or pipeline diagnostics to verify code edits pass before declaring success or completing a task.
