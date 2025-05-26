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

> **Remote storage is not configured at the moment.**  
> Integration with Google Drive was attempted but encountered known [issues](https://dvc.org/doc/user-guide/data-management/remote-storage/google-drive) .  
> For now, **DVC stores all outputs locally**.

### Prerequisites

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### Running the full pipeline

To run all stages:

```bash
dvc init
dvc repro
```

This command will:
- Download and preprocess the dataset
- Generate bag-of-words features
- Train the sentiment analysis model
- Evaluate the model 

Future features:
- Produce metrics
- Generate plots for performance visualization

You may, at this stage, encounter this issue:
```bash
ERROR: failed to reproduce 'featurize':  output 'models/bow_sentiment_model.pkl' is already tracked by SCM (e.g. Git)
```

This is due to the fact that we did not set up a remote storage for DVC yet. Follow the instructions in the terminal to solve it.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         model-training and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
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
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

