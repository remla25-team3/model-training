# model-training

<p>
    <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
        <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
    </a>
    <img src="https://github.com/remla25-team3/model-training/raw/badge-badges/coverage.svg" />
    <img src="https://github.com/remla25-team3/model-training/raw/badge-badges/pylint.svg" />
    <img src="https://github.com/remla25-team3/model-training/raw/badge-badges/ml_test_score.svg" />
</p>

This repository contains the **machine learning training pipeline** according to instructions in the [Restaurant Sentiment Analysis](https://github.com/proksch/restaurant-sentiment) project.

Structured according to the [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) template.

> **Note**: The `models/` and `data/` directories are not stored in the Git repository.  
> They are tracked and versioned using [DVC](https://dvc.org) and can be pulled from remote storage (see below).

### 🔧 Prerequisites

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the ML Pipeline (with DVC)

This project uses [DVC](https://dvc.org) to define and manage the ML pipeline, including:

- Data download and preprocessing
- Feature extraction
- Model training and evaluation
- Model and data versioning

### 📁 DVC Remote Access Setup (via Service Account)

This project uses a **shared Google Drive folder** as a [DVC remote storage](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive). To **download** or **upload** datasets and models, follow the steps below:

#### 🔐 1. Set Up a Custom Google Cloud Project

Follow **Steps 1–3** of the official DVC guide here: [Using a Custom Google Cloud Project](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

#### 🛂 2. Create a Service Account

Now follow **Step 1** of this section: [Using Service Accounts](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-service-accounts)

- After creating the service account, download the `.json` key file.
- Place this file in the **root of the repository**, and name it:  
  ```text
  service_key.json
  ```

#### 🔗 3. Grant Access to the Shared Drive

- Open the `.json` file and locate the `client_email` entry.
- Copy the email address.
- Visit the [shared Google Drive folder](https://drive.google.com/drive/folders/1n5l1DxOWcoMcQXKRBHFJO4iDftQTHAgc?usp=sharing) and **share** the folder with that email address (as an editor).

✅ You're all set! You can now interact with the DVC remote:

```bash
dvc pull  # to fetch data/models from the remote
dvc push  # to upload updates to the remote (only if you are a member of team 3)
```

### 🔁 Reproducing the Full Pipeline

To reproduce all pipeline stages:

```bash
dvc repro
```

DVC will only re-run the stages affected by changes to code, data, or parameters.

#### Common use cases:
- You modified model code or preprocessing steps.
- You adjusted hyperparameters.
- You’re working in a new environment and want to reproduce all results.

If you're a **Team 3 member** and made changes that should be shared, run:

```bash
dvc repro
dvc push
```

### Running and Comparing Experiments

To run experiments:

```bash
dvc exp run
```

To explore experiment results:

```bash
dvc exp show
```

> Press `Q` to exit the table viewer.

---

## ✅ Tests

This repository includes a comprehensive test suite located in the `tests/` directory, organized according to the [ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) methodology.

The test structure is as follows:

- `tests/data/`: Validates **feature and data integrity**, including format checks, input consistency...
- `tests/development/`: Focuses on **model development** and robustness, metamorphic tests, data slices...
- `tests/infrastructure/`: Checks **ML infrastructure** components for correctness, reproducibility, latency bounds, inference determinism...
- `tests/monitoring/`: Ensures **monitoring** and drift detection, such as model staleness and prediction consistency under perturbation.

> More information about the intent and scope of each test is included in the docstring at the beginning of every test file

### 🧪 Running Tests

To run all tests with coverage analysis, execute the following from the root directory:

```bash
pytest
```

This automatically:
- Runs the full test suite
- Computes coverage
- Evaluates test adequacy based on the ML Test Score

#### 📊 Viewing Test Coverage

Coverage results will be automatically printed in the terminal. Additionally, an HTML report will be generated in the `htmlcov/` directory. You can open it with:

```bash
xdg-open htmlcov/index.html  # For Linux
# or
open htmlcov/index.html      # For macOS
```

#### 🧠 About Test Adequacy

The ML Test Score evaluates how comprehensively our tests cover critical ML production concerns, based on Google's [ML Test score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/).
- Each test category (data, model, infrastructure, monitoring) defines a set of relevant keywords (e.g., "synonym", "robustness", "latency")
- The adequacy score is based on the presence of these keywords in test files
- Each category contributes to a weighted global score out of 100

The score breakdown is printed in the terminal after tests complete, and saved to metrics/ml_test_score.json.

---

## 🧹 Code Quality & Linting

To maintain a high standard of code quality, this repository uses the following tools:
- **Pylint** for deep static analysis, naming conventions, and ML-specific smells
- **Flake8** for fast PEP8-compliant style checks
- **Bandit** for detecting potential security vulnerabilities

These tools are configured specifically for the REMLA 2025 project and are applied to **all source code and test code** in this repository, because we believe that tests should follow the same best practices as production code to maintain clarity, reliability, and maintainability.

### 🔍 Pylint

We use a custom **pylint** configuration that includes, among other personalizations, a custom plugin to detect the [Randomness Uncontrolled Code Smell](https://hynn01.github.io/ml-smells/posts/codesmells/14-randomness-uncontrolled/).

To run it:
```bash
pylint model_training tests scripts
```

To test if our custom plugin actually detects the code smell, we implemented a simple test in which we use **random**, **numpy.random** and **tensorflow.random** without specifying the **random seed**.

To see that our plugin works, run: 
```bash
pylint pylint_custom/tests/test_randomness.py
```

Expected output:
```bash

pylint_custom/tests/test_randomness.py:1:0: W9002: No random seed set for module(s): numpy, random, tensorflow (missing-random-seed)
```

### 🔍 Flake8

Used for fast style and formatting checks, ensuring consistent code style and readability across the project.

To run it:
```bash
flake8 model_training tests scripts
```

Expected output:
```bash
0
```

### 🔍 Bandit

Bandit scans for common Python security issues, ensuring safe and robust code practices.

To run it:
```bash
 bandit -r model_training tests scripts --config bandit.yml
 ```

Expected output:
 ```bash
[main]  INFO    profile include tests: None
[main]  INFO    profile exclude tests: B301,B311,B101
[main]  INFO    cli include tests: None
[main]  INFO    cli exclude tests: None
[main]  INFO    using config: bandit.yml
[main]  INFO    running on Python 3.12.5
Run started:2025-06-05 15:18:04.074228

Test results:
        No issues identified.

Code scanned:
        Total lines of code: 1156
        Total lines skipped (#nosec): 0

Run metrics:
        Total issues (by severity):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 0
        Total issues (by confidence):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 0
Files skipped (0):
```

### 🔁 Continuous Integration

All tests and linters (**pylint**, **flake8**,**bandit**) are integrated into the GitHub Actions workflow. Test failures will fail the CI build. Test adequacy and coverage metrics are tracked via **coverage.py** and reported directly in the CI logs.


## ✅ Project Quality Metrics

![Coverage](https://github.com/remla25-team3/model-training/raw/badge-badges/coverage.svg)
![Pylint](https://github.com/remla25-team3/model-training/raw/badge-badges/pylint.svg)
![ML Test Score](https://github.com/remla25-team3/model-training/raw/badge-badges/ml_test_score.svg)

These badges are automatically updated via GitHub Actions on every push and pull request.  
They reflect:

- ✅ Test coverage percentage (`pytest-cov`)
- ✅ Lint quality score (`pylint`)
- ✅ ML Test Score adequacy (based on Google's ML Test Score)

---

## Project Organization

```
├── LICENSE                  <- MIT License for open-source distribution
├── Makefile                <- Common workflow commands like `make data` or `make train`
├── README.md               <- Top-level project documentation
│
├── data                    <- Tracked by DVC: all data sources and preprocessed forms
│   ├── external            <- Raw source data (e.g., TSV dump)
│   ├── interim             <- Preprocessed but not yet feature-engineered data
│   └── processed           <- Final feature matrix and test split (features.csv, X_test.csv, y_test.csv)
│
├── dvc.yaml                <- DVC pipeline definition
├── dvc.lock                <- DVC pipeline state and hashes
│
├── model_training          <- Core source code for model development and evaluation
│   ├── __init__.py
│   ├── config.py           <- Path management and config constants
│   ├── dataset.py          <- Dataset loading and preprocessing logic
│   ├── features.py         <- Feature engineering steps
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── train.py        <- Training routine and model saving
│   │   ├── evaluate.py     <- Evaluation metrics and pipeline
│   │   └── predict.py      <- Inference logic for predictions
│
├── output                  <- Model evaluation output (e.g., DVC metrics.json)
│
├── pylint_custom           <- Custom Pylint checkers for ML-specific code smells
│   ├── __init__.py
│   ├── ml_pylint.py        <- Custom Code Smell checker implementation (Uncontrolled Randomness)
│   └── tests
│       ├── __init__.py
│       └── test_randomness.py  <- Tests for the randomness checker
│
├── requirements.txt        <- Python dependencies
├── pyproject.toml          <- Project metadata and tool configuration
|
├── .pylint                 <- pylint linter configuration
├── .flake8                 <- flake8 linter configuration
├── bandit.yml              <- bandit linter configuration
|
├── pytest.ini              <- Pytest configuration
├── setup.cfg               <- Tooling configuration
│
├── scripts
│   └── ml_test_score.py    <- Utility script to calculate test adequacy based on Google's ML Test Score
│
├── metrics
│   └── ml_test_score.json  <- Test adequacy score, based on scripts/ml_test_score calculation
|
├── references
│   └── README.md           <- Any reference materials, citations, or external notes
│
├── release-please-config.json <- Config for GitHub Release automation
├── htmlcov                 <- Generated HTML test coverage reports
|
│
└── tests                   <- Full test suite, categorized by ML Test Score themes
    ├── conftest.py
    ├── data
    │   └── test_data.py
    ├── development
    │   ├── test_metamorphic.py
    │   └── test_model.py
    ├── infrastructure
    │   └── test_infrastructure.py
    └── monitoring
        └── test_monitoring.py

```

--------
