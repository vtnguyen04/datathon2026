# Datathon 2026: The Gridbreaker
> Predictive and prescriptive analytics pipeline for Vietnamese Fashion E-Commerce.

This repository contains the source code for our Datathon 2026 submission. The solution encompasses an automated EDA pipeline, a Multiple Choice Question (MCQ) evaluation solver, and a 3-tier ensemble forecasting architecture capable of modeling complex multiplicative seasonality and structural market shifts.

## Installation

This project uses `uv` for ultra-fast, deterministic Python dependency management.

**OS X & Linux:**
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/vtnguyen04/datathon2026.git
cd datathon2026

# Create a virtual environment and sync exact dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

**Windows:**
```powershell
# Install uv (if not already installed)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Clone the repository
git clone https://github.com/vtnguyen04/datathon2026.git
cd datathon2026

# Create a virtual environment and sync exact dependencies
uv sync

# Activate the virtual environment
.venv\Scripts\activate
```

## Dataset Configuration

Before executing the pipelines, ensure the 12 original competition CSV files are placed inside the `data/raw/` directory:
```text
datathon2026/
└── data/
    └── raw/
        ├── customers.csv
        ├── orders.csv
        ├── sales.csv
        └── ...
```

## Usage

The project is structured into three independent workflows corresponding to the Datathon phases.

### Part 1: MCQ Automation
Extracts answers for the MCQ section directly from the raw data to prevent human bias.
```bash
uv run python run_mcq.py
```

### Part 2: Exploratory Data Analysis (EDA)
Generates high-resolution visualizations and diagnostic metrics used in the technical report.
```bash
uv run python run_eda.py
```

### Part 3: Revenue & COGS Forecasting
Executes the final forecasting engine. This script performs feature engineering, trains the LightGBM/Ridge ensemble, and generates the final submission file.
```bash
uv run python run_forecasting.py
```
Generated submission files will be automatically saved in the `data/submissions/` directory.

## Repository Structure

```text
.
├── data/
│   ├── raw/                 # Competition datasets (git-ignored)
│   └── submissions/         # Final generated outputs
├── src/                     # Core analytical library
│   ├── calibration/         # Post-processing and scaling modules
│   ├── core/                # Configuration and logging
│   ├── data/                # Data loaders and schema validators
│   ├── ensemble/            # Model blending and stacking methods
│   ├── features/            # Feature engineering (Temporal, Fourier, DOW)
│   ├── models/              # LightGBM, XGBoost, and Ridge implementations
│   └── pipelines/           # Execution logic for the runner scripts
├── run_mcq.py               # Part 1 entry point
├── run_eda.py               # Part 2 entry point
├── run_forecasting.py       # Part 3 entry point
├── pyproject.toml           # Project metadata and dependencies
└── uv.lock                  # Deterministic dependency lockfile
```

## Authors

* **Do Hoang Khai** 
* **Nguyen Vo Thanh**
* **Hoang Huynh Huy**
* **Nghia Vo Minh**

## License

Distributed under the MIT License.
