# DL lab

Small, standalone deep learning and machine learning examples.

## Setup

Use Python 3.12 for the broadest compatibility with the TensorFlow example. Create an isolated environment and install the dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Word2Vec embeddings

`word2vec_embeddings.py` trains a small Word2Vec model on a three-sentence sample corpus, prints the embedding for `learning`, and lists similar words. A fixed seed makes the example reproducible.

```bash
python word2vec_embeddings.py
```

## Iris deep neural network

`iris_dnn_classifier.py` standardizes the Iris measurements using statistics learned from the training split, trains a dense neural network to classify the three Iris species, and prints test accuracy alongside predicted and actual class indices. The split and random seeds are fixed for repeatable runs.

```bash
python iris_dnn_classifier.py
```

## MNIST convolutional neural network

`mnist_cnn_classifier.py` trains a convolutional neural network on handwritten digit images, evaluates it on the MNIST test set, and prints predictions for the first ten test images. TensorFlow downloads the MNIST data on the first run.

```bash
python mnist_cnn_classifier.py
```

## MNIST autoencoder

`mnist_autoencoder.py` trains a small autoencoder to compress each 28 × 28 digit image into 32 values and reconstruct it. It plots five original/reconstructed image pairs and prints the original and compressed array shapes. TensorFlow downloads MNIST on the first run.

```bash
python mnist_autoencoder.py
```
