"""Train and evaluate a dense neural network on the Iris dataset."""

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input


def main() -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    iris = load_iris()
    features, labels = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = Sequential(
        [
            Input(shape=(4,)),
            Dense(16, activation="relu"),
            Dense(12, activation="relu"),
            Dense(8, activation="relu"),
            Dense(3, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    probabilities = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(probabilities, axis=1)
    print("\nPredicted classes:", predicted_classes)
    print("Actual classes:   ", y_test)


if __name__ == "__main__":
    main()
