# model-training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Contains the ML training pipeline. The resulting model is to be accessed through a public link by `model-service`.

Structured according to the [Cookiecutter Data Science](https://github.com/drivendataorg/cookiecutter-data-science) template.

## Features

- Trains a sentiment analysis model according to instructions in the [Restaurant Sentiment Analysis](https://github.com/proksch/restaurant-sentiment) project.
- The model is stored and retrievable at `model_training/data/sentiment_model.pkl`.

## Usage

To use the trained model in another service (i.e. `model-service`), download it from:

https://github.com/remla25-team3/model-training/tree/main/model_training/data/sentiment_model.pkl

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

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
|
├── data                (managed my DVC)
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│
├── models             <- (Managed by DVC) Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         model-training and configuration for tools like black
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── model-training   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes model-training a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models  
    |   ├── evaluate.py         <- Code to evaluate model performance (accuracy, f1, precision, recall, ROC-AUC)
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------


## ✅ Project Quality Metrics

![Coverage](https://github.com/remla25-team7/model-training/raw/badge-badges/coverage.svg)
![Pylint](https://github.com/remla25-team7/model-training/raw/badge-badges/pylint.svg)
![ML Test Score](https://github.com/remla25-team7/model-training/raw/badge-badges/ml_test_score.svg)

These badges are automatically updated via GitHub Actions on every push and pull request.  
They reflect:

- ✅ Test coverage percentage (`pytest-cov`)
- ✅ Lint quality score (`pylint`)
- ✅ ML Test Score adequacy (based on Google's ML Test Score)

