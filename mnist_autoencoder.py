"""Train an MNIST autoencoder and compare original and reconstructed images."""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers


def main() -> None:
    tf.random.set_seed(42)

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
    x_test = x_test.astype("float32").reshape(-1, 784) / 255.0

    input_image = layers.Input(shape=(784,))
    encoded = layers.Dense(32, activation="relu")(input_image)
    decoded = layers.Dense(784, activation="sigmoid")(encoded)
    autoencoder = Model(input_image, decoded, name="mnist_autoencoder")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    autoencoder.fit(
        x_train,
        x_train,
        epochs=5,
        batch_size=256,
        shuffle=True,
        validation_data=(x_test, x_test),
    )

    encoder = Model(input_image, encoded, name="mnist_encoder")
    encoded_images = encoder.predict(x_test, verbose=0)
    reconstructed_images = autoencoder.predict(x_test, verbose=0)

    display_count = 5
    figure, axes = plt.subplots(2, display_count, figsize=(10, 4))
    for index in range(display_count):
        axes[0, index].imshow(x_test[index].reshape(28, 28), cmap="gray")
        axes[0, index].set_title("Original")
        axes[1, index].imshow(
            reconstructed_images[index].reshape(28, 28), cmap="gray"
        )
        axes[1, index].set_title("Reconstructed")

        for axis in axes[:, index]:
            axis.axis("off")

    figure.tight_layout()
    plt.show()

    print("Original size:", x_test.shape)
    print("Compressed size:", encoded_images.shape)


if __name__ == "__main__":
    main()
