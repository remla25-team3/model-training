# model-training

Contains the ML training pipeline.
The resulting model is to be accessed through a public link by `model-service`.

## Features

- Trains a sentiment analysis model according to instructions in the [Restaurant Sentiment Analysis](https://github.com/proksch/restaurant-sentiment) project.
- The model is stored and retrievable at `model_training/data/sentiment_model.pkl`.

## Usage

To use the trained model in another service (i.e. `model-service`), download it from:

https://github.com/remla25-team3/model-training/tree/main/model_training/data/sentiment_model.pkl
