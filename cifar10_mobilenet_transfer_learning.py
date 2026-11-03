"""Classify CIFAR-10 images using a frozen ImageNet MobileNetV2 base."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input


IMAGE_SIZE = (96, 96)
BATCH_SIZE = 32
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def make_dataset(images, labels, *, training: bool) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(10_000, seed=42, reshuffle_each_iteration=True)

    def preprocess(image, label):
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        return image, label

    return dataset.batch(BATCH_SIZE).map(
        preprocess, num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    tf.random.set_seed(42)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Hold out the last 20% of the provided training set for validation.
    validation_start = int(len(x_train) * 0.8)
    train_dataset = make_dataset(
        x_train[:validation_start], y_train[:validation_start], training=True
    )
    validation_dataset = make_dataset(
        x_train[validation_start:], y_train[validation_start:], training=False
    )
    test_dataset = make_dataset(x_test, y_test, training=False)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = Sequential(
        [
            Input(shape=(*IMAGE_SIZE, 3)),
            base_model,
            GlobalAveragePooling2D(),
            Dense(64, activation="relu"),
            Dense(len(CLASS_NAMES), activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    history = model.fit(
        train_dataset,
        epochs=5,
        validation_data=validation_dataset,
    )

    loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    image_batch, label_batch = next(iter(test_dataset))
    probabilities = model.predict(image_batch[:1], verbose=0)[0]
    predicted_class = int(np.argmax(probabilities))
    actual_class = int(label_batch[0].numpy())
    print("Actual:", CLASS_NAMES[actual_class])
    print("Predicted:", CLASS_NAMES[predicted_class])

    print(f"Training accuracy: {history.history['accuracy'][-1] * 100:.2f}%")
    print(
        "Validation accuracy: "
        f"{history.history['val_accuracy'][-1] * 100:.2f}%"
    )

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
