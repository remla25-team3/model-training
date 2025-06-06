# model-training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Contains the ML training pipeline. The resulting model is to be accessed through a public link by `model-service`.

Structured according to the [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) template.

## Features

- Trains a sentiment analysis model according to instructions in the [Restaurant Sentiment Analysis](https://github.com/proksch/restaurant-sentiment) project.
- The model is stored and retrievable via DVC from the remote storage at `models/sentiment_model.pkl`.

> **Note**: The `models/` and `data/` directories are not stored in the Git repository.  
> They are tracked and versioned using [DVC](https://dvc.org) and can be pulled from remote storage.

## Running the ML Pipeline (with DVC)

This project uses [DVC](https://dvc.org) to define and manage the ML pipeline, including:

- Data download and preprocessing
- Feature extraction
- Model training and evaluation

### Prerequisites

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### DVC Remote Access Setup

This project uses a shared Google Drive folder as a DVC remote.
In order to download (pull) or upload (push) data, you’ll need to authenticate using your Google account via OAuth.

To avoid errors such as “This app is blocked,” we recommend that each user create their own Google Cloud OAuth credentials.

Follow the DVC documentation guide until step 6 to create your credentials:
[Using a Custom Google Cloud project](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive#using-a-custom-google-cloud-project-recommended)

> **Note**: In Step 5, make sure to select **Desktop app**

There's one last step to do: navigate to [Google Auth Platform -> Audience](https://console.cloud.google.com/auth/audience) and under testing press **Publish app** to be able to access it.

Once you've generated your client_id and client_secret, configure them by running:

```bash
dvc remote modify gdrive_remote gdrive_client_id 'YOUR_CLIENT_ID'
dvc remote modify gdrive_remote gdrive_client_secret 'YOUR_CLIENT_SECRET'
```

> **Note:** Do not push the Google Drive credentials to GitHub.  
> After testing or setting up the remote, either save the last two lines of `.dvc/config` somewhere safe, or remove them before committing (you can always go get them in you profile):
>
> ```ini
> [core]
>    remote = gdrive_remote
> ['remote "gdrive_remote"']
>    url = gdrive://1n5l1DxOWcoMcQXKRBHFJO4iDftQTHAgc
> ```
>
> You’ll need to re-add them locally each time to interact with the remote storage.

After setup, you can use the following commands to synchronize data with the remote storage:

```bash
dvc pull
```

The first time you will need to identify (log in with the same Google account you followed the previous steps with), Google may then recognize the access as not safe, but proceed anyway (under Advanced > Go on)

Well done! Now you should be able to interact with the remote storage!

## How to reproduce the full pipeline

To execute all steps in the pipeline, use:
```bash
dvc repro
```
This command automatically detects changes in your code, data, or parameters, and only reruns the pipeline stages that are affected.
Common scenarios where you’ll use this:
- You changed model hyperparameters
- You modified preprocessing or model code
- You’re working on a new environment or system and need to reproduce results from scratch

## ✅ Tests

This repository includes a comprehensive test suite located in the `tests/` directory, organized according to the [ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) methodology. The test structure is as follows:

- `tests/data/`: Validates **feature and data integrity**, including format checks and input consistency.
- `tests/development/`: Covers **model development**, including:
  - Functional tests
  - Metamorphic testing (e.g., synonym substitution) to assess model robustness.
  - Data slice testing for subpopulation performance.
- `tests/infrastructure/`: Checks **ML infrastructure** components for correctness and reproducibility.
- `tests/monitoring/`: Validates that metrics, logging, and monitoring hooks are correctly configured and functioning.

### Running Tests

To run all tests with coverage analysis, execute the following from the root directory:

```bash
pytest
```

### Viewing Coverage

Coverage results will be printed in the terminal. Additionally, an HTML report will be generated in the `htmlcov/` directory. You can open it with:

```bash
xdg-open htmlcov/index.html  # For Linux
# or
open htmlcov/index.html      # For macOS
```

### Continuous Integration

To ensure code quality, security, and readability across our Python codebase, we use a combination of three linters: **Pylint**, **Flake8**, and **Bandit**. These tools are configured specifically for the REMLA project and are applied to **all source code and test code**, because we believe that tests should follow the same best practices as production code to maintain clarity, reliability, and maintainability.


#### Pylint

**Purpose**: Full-featured static analyzer to detect bugs, code smells, and maintainability issues.

**Configuration highlights**:
- Maximum line length set to `110`.
- Custom plugin `pylint_custom.ml_pylint` adds rules for ML-specific code smell: missing random seeds.
- Enables built-in plugins such as `check_elif`, `bad_builtin`, `docparams`, `set_membership`, `typing`, etc.
- Disables rules that conflict with ML code style, such as `too-few-public-methods`, `too-many-arguments`, and personalizes variable naming constraints (e.g., `X`, `y`, `df` are allowed).

**Run it**:
```bash
pylint model_training tests scripts pylint_custom
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

#### Flake8

**Purpose**: Lightweight syntax and style checker enforcing PEP8 compliance.

**Configuration highlights**:
- Ignores the E402 (module-level import not at top) rule
- Maximum line length: 110
- Excludes common folders and artifacts: .dvc, .pytest_cache, .venv, models, htmlcov, output, etc.

**Run it**:
```bash
flake8 model_training tests scripts pylint_custom
```
Expected output:
```bash
0
```

#### Bandit

Purpose: Security-focused linter that scans for common Python vulnerabilities and unsafe practices.

Configuration highlights:
- Issue aggregation is enabled per file to improve readability (`aggregated: true`).
- Skipped checks:
  - `B101`: Use of `assert`. Assertions are permitted in our tests and infrastructure checks.
  - `B311`: Use of `random` without seed. We skip this as it is already enforced via our custom Pylint plugin.
- Severity threshold is set to `LOW`, ensuring even minor issues are caught.
- Confidence threshold is set to `MEDIUM`, balancing coverage and precision.
- Verbose output is enabled to show lines of offending code (`show-issue-lines: true`).
- Excluded folders include non-source and generated content like `.venv/`, `.dvc/`, `.git/`, `models/`, `output/`, and similar.

**Run it**:
```bash
 bandit -r model_training scripts pylint_custom tests --config bandit.yml
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

### Continuous Integration

All tests and linters (**pylint(**, (**flake8(**, (**bandit(**) are integrated into the GitHub Actions workflow. Test failures or critical linter violations will fail the CI build. Test adequacy and coverage metrics are tracked via **coverage.py** and reported directly in the CI logs.






## ✅ Project Quality Metrics

![Coverage](https://github.com/remla25-team3/model-training/raw/badge-badges/coverage.svg)
![Pylint](https://github.com/remla25-team3/model-training/raw/badge-badges/pylint.svg)
![ML Test Score](https://github.com/remla25-team3/model-training/raw/badge-badges/ml_test_score.svg)

These badges are automatically updated via GitHub Actions on every push and pull request.  
They reflect:

- ✅ Test coverage percentage (`pytest-cov`)
- ✅ Lint quality score (`pylint`)
- ✅ ML Test Score adequacy (based on Google's ML Test Score)

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
│   └── plots.py            <- Optional visualization utilities
│
├── output                  <- Model evaluation output (e.g., metrics.json)
│
├── pylint_custom           <- Custom Pylint checkers for ML-specific code smells
│   ├── __init__.py
│   ├── ml_pylint.py        <- Custom checker implementation
│   └── tests
│       ├── __init__.py
│       └── test_randomness.py  <- Tests for the randomness checker
│
├── pyproject.toml          <- Project metadata and tool configuration
├── pytest.ini              <- Pytest configuration
├── requirements.txt        <- Python dependencies
├── setup.cfg               <- Linter/tooling configuration
│
├── scripts
│   └── update_readme.py    <- Utility script to programmatically update the README
│
├── references
│   └── README.md           <- Any reference materials, citations, or external notes
│
├── release-please-config.json <- Config for GitHub Release automation
│
├── tests                   <- Full test suite, categorized by ML Test Score themes
│   ├── conftest.py
│   ├── data
│   │   └── test_data.py
│   ├── development
│   │   ├── test_data_slices.py
│   │   ├── test_metamorphic.py
│   │   └── test_model.py
│   ├── infrastructure
│   │   └── test_infrastructure.py
│   ├── monitoring
│   │   └── test_monitoring.py
│   └── htmlcov             <- Generated HTML test coverage reports
```

--------