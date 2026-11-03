"""Train a convolutional neural network to classify MNIST digits."""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D
from tensorflow.keras.utils import to_categorical


def main() -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    model = Sequential(
        [
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    probabilities = model.predict(x_test[:10], verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    actual_classes = np.argmax(y_test[:10], axis=1)
    print("Predicted classes:", predicted_classes)
    print("Actual classes:   ", actual_classes)


if __name__ == "__main__":
    main()
