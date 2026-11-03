"""Forecast a synthetic time series with an LSTM network."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, LSTM


SEQ_LEN = 20
TRAIN_FRACTION = 0.8


def create_sequences(dataset: np.ndarray, seq_length: int) -> tuple[np.ndarray, np.ndarray]:
    features, targets = [], []
    for index in range(len(dataset) - seq_length):
        features.append(dataset[index : index + seq_length, 0])
        targets.append(dataset[index + seq_length, 0])
    return np.asarray(features), np.asarray(targets)


def main() -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    time = np.arange(0, 200, 0.1)
    data = np.sin(time) + 0.1 * np.random.randn(len(time))
    train_end = int(len(data) * TRAIN_FRACTION)

    plt.figure(figsize=(10, 4))
    plt.plot(time, data)
    plt.title("Synthetic Time Series Data")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.tight_layout()
    plt.show()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[:train_end].reshape(-1, 1))
    data_scaled = scaler.transform(data.reshape(-1, 1))

    X, y = create_sequences(data_scaled, SEQ_LEN)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Targets before train_end belong to training; later targets are held out.
    split = train_end - SEQ_LEN
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential(
        [
            Input(shape=(SEQ_LEN, 1)),
            LSTM(50, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse")
    model.summary()
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_test, y_test),
        shuffle=False,
        verbose=1,
    )

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.tight_layout()
    plt.show()

    predictions = model.predict(X_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(10, 4))
    plt.plot(y_test_actual, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.title("LSTM Forecast on Held-Out Data")
    plt.xlabel("Test time step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
