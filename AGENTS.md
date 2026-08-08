# Agent Onboarding & Project-Level Guide (`AGENTS.md`)

Welcome! This document serves as the onboarding and project guide for AI agents and developers working on this repository. It defines our environment configuration, project architecture, mathematical modeling practices, code style standards, and testing procedures.

---

## 1. Setup Commands

Follow these steps to set up the Python virtual environment, install dependencies, and configure the prescriptive Optimization solvers.

### Virtual Environment Setup & Dependency Installation
We recommend using Python 3.12. You can create a virtual environment and install the required dependencies using standard `pip`:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip and install standard dependencies
pip install --upgrade pip
pip install -r requirements.txt pytest
```

### Prescriptive Solver Installation
Our Optimization engine uses open-source solvers (GLPK and COIN-OR CBC) to solve prescriptive linear and mixed-integer programming formulations. Install them using your system package manager:

#### On Debian/Ubuntu Systems (`apt`):
```bash
sudo apt-get update
sudo apt-get install -y glpk-utils coinor-cbc
```

#### On macOS Systems (`Homebrew`):
```bash
brew install glpk cbc
```

#### Verification:
Verify that the solvers are successfully installed and available in your system path:
```bash
glpsol --version
cbc --version
```

---

## 2. Project Overview

This repository is a **Decision Intelligence (DI)** platform that integrates predictive and prescriptive models. Our pipeline is designed to enable data-driven planning and decision-making by combining two distinct methodologies:

1. **Machine Learning Predictions (Predictive):**
   - High-performance regressors and classifiers (e.g., XGBoost, CatBoost, LightGBM).
   - High-regularization linear models (e.g., LassoCV, RidgeCV, ElasticNetCV) customized for small-dataset robustness (Ames Housing, <1,500 rows).
   - Core pipeline processes target transformations, quality ordinal encodings, and neighborhood target rank mapping to minimize cross-validation (CV) to leaderboard (LB) divergence.

2. **Operations Research & Mathematical Programming (Prescriptive):**
   - Mathematical formulations (built with **Pyomo** and **SciPy SLSQP**) to optimize resource allocations, ensemble weights, and business decisions under constraints.
   - 0.1-second vectorized non-negative least squares optimizer utilizing `scipy.optimize.minimize`.

---

## 3. Mathematical Modeling Conventions

When defining or modifying prescriptive Optimization formulations (such as **Pyomo** concrete or abstract models), future development sessions must strictly adhere to these modeling conventions:

* **Define Pyomo Parameters with `mutable=True`**:
  All Pyomo parameters must be declared with `mutable=True`. This enables rapid parametric sensitivity analysis, allowing the model to be repeatedly re-solved with modified parameter values without needing to rebuild the entire constraint matrix.

  *Example:*
  ```python
  model.unit_cost = Param(initialize=12.5, mutable=True, doc="Unit production cost of item")
  ```

* **Attach Detailed `doc="..."` Strings**:
  To ensure future sessions can read, interpret, and logically verify the model, attach a descriptive `doc` string to every single Pyomo `Var`, `Param`, `Constraint`, and `Objective`.

  *Example:*
  ```python
  model.x = Var(within=NonNegativeReals, doc="Quantity of product A produced")
  model.production_limit = Constraint(expr=model.x <= 100, doc="Maximum capacity limit for product A")
  ```

---

## 4. Code Style & Python Standards

We maintain a high standard of code readability, maintainability, and clean architecture.

* **Strict Type Hints**:
  All Python modules must employ comprehensive and strict type hints (PEP 484) for all function arguments and return signatures.

  *Example:*
  ```python
  def calculate_margin(revenue: float, cost: float) -> float:
      return revenue - cost
  ```

* **Separation of Concerns**:
  Keep mathematical formulations and business logic strictly decoupled from visualization and user interface logic.
  - **Do NOT** place Pyomo models, SciPy optimization formulas, or ML preprocessing logic inside Streamlit or CLI views.
  - Place core algorithms, pipelines, and models in `src/` modules, and import them into UI components (like Streamlit applications) if UI logic is present.

* **Linting and Static Analysis**:
  Run `ruff` or static analysis tools before submitting work to ensure zero formatting or style discrepancies:
  ```bash
  python3 -m ruff check src/
  ```

---

## 5. Testing & Verification

A robust automated test suite is critical to preventing regressions across our predictive pipelines and prescriptive solvers.

* **How to Run Unit Tests**:
  We use `pytest` for automated test execution. Always verify your changes before submitting code by running:
  ```bash
  PYTHONPATH=. pytest tests/ -v
  ```

* **Integration of Pre-Commit and PR Checks**:
  Before requesting a code review or submitting a Pull Request:
  1. Ensure the pytest suite completes with **100% success**.
  2. If any test fails, diagnose the root cause immediately, fix the implementation, and rerun tests.
  3. Never commit unfinished or breaking test modifications to the main branch.

---

## 6. Automated Submissions with Kaggle API

To automate submission upload and leaderboard fetching from a cloud VM or any CLI session, install and configure the Kaggle API package (`kaggle>=2.0.0`) securely.

### Environment Variable Injection (Recommended)
Set the following environment variables in your terminal session before executing any commands:
```bash
export KAGGLE_USERNAME="aliazizi1"
export KAGGLE_API_TOKEN="KGAT_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Automation Commands:
Once authenticated, you can submit predictions and list results directly:
```bash
# Submit the final submission
kaggle competitions submit -c house-prices-advanced-regression-techniques -f submission.csv -m "Jules Cloud VM: Leak-free SLSQP Stacking Ensemble"

# Fetch latest leaderboard submissions and scores
kaggle competitions submissions -c house-prices-advanced-regression-techniques
```
