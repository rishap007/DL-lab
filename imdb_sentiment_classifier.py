"""Classify IMDB movie reviews as positive or negative."""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences


VOCABULARY_SIZE = 10_000
REVIEW_LENGTH = 200


def main() -> None:
    tf.random.set_seed(42)

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(
        num_words=VOCABULARY_SIZE
    )
    x_train = pad_sequences(x_train, maxlen=REVIEW_LENGTH)
    x_test = pad_sequences(x_test, maxlen=REVIEW_LENGTH)

    model = Sequential(
        [
            Embedding(VOCABULARY_SIZE, 32),
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
    )

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")

    probabilities = model.predict(x_test[:5], verbose=0).ravel()
    print("\nReview classification:")
    for index, probability in enumerate(probabilities, start=1):
        sentiment = "Positive" if probability >= 0.5 else "Negative"
        print(f"Review {index}: {sentiment}")
        print(f"Probability: {probability:.4f}")


if __name__ == "__main__":
    main()
