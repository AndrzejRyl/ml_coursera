import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.image import resize
import numpy as np


def load_data():
    # Load the MNIST dataset
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Filter the images and labels to include only those with labels 0 or 1
    train_filter_mask = (train_labels == 0) | (train_labels == 1)
    filtered_train_images = train_images[train_filter_mask]
    filtered_train_labels = train_labels[train_filter_mask]

    test_filter_mask = (test_labels == 0) | (test_labels == 1)
    filtered_test_images = test_images[test_filter_mask]
    filtered_test_labels = test_labels[test_filter_mask]

    # Expand dimensions to add a channel (from 28x28 to 28x28x1)
    train_images_expanded = np.expand_dims(filtered_train_images, axis=-1)
    train_labels_expanded = np.expand_dims(filtered_train_labels, axis=-1)
    test_images_expanded = np.expand_dims(filtered_test_images, axis=-1)
    test_labels_expanded = np.expand_dims(filtered_test_labels, axis=-1)

    # Resize images to 20x20
    train_images_resized = np.array([resize(image, [20, 20]).numpy() for image in train_images_expanded])
    test_images_resized = np.array([resize(image, [20, 20]).numpy() for image in test_images_expanded])

    # Flatten the resized images to 400-element arrays
    train_images_flattened = train_images_resized.reshape(-1, 400)
    test_images_flattened = test_images_resized.reshape(-1, 400)

    return train_images_flattened, train_labels_expanded, test_images_flattened, test_labels_expanded
