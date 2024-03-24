import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Sequential

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)


def load_linear_regression_data():
    x_train = np.array([[1.0], [2.0]], dtype=np.float32)  # (size in 1000 square feet)
    y_train = np.array([[300.0], [500.0]], dtype=np.float32)  # (price in 1000s of dollars)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_train, y_train, marker='x', c='r', label="Data Points")
    ax.legend(fontsize='xx-large')
    ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    ax.set_title("Linear regression inputs")
    plt.show()

    return x_train, y_train


def plot_linear_regression(x_train, y_train):
    # Create linear layer
    linear_layer = Dense(units=1, activation='linear', )

    # Activate layer initializing weights
    linear_layer(x_train[0].reshape(1, 1))

    # Set pre-trained weights
    set_w = np.array([[200]])
    set_b = np.array([100])
    linear_layer.set_weights([set_w, set_b])

    # Calculate predictions based on input data and pretrained weights
    prediction_tf = linear_layer(x_train)

    # Plot
    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_train, y_train, marker='x', c='r', label="Data Points")
    ax.plot(x_train, prediction_tf, c='b', label="Prediction")
    ax.legend(fontsize='xx-large')
    ax.set_ylabel('Price (in 1000s of dollars)', fontsize='xx-large')
    ax.set_xlabel('Size (1000 sqft)', fontsize='xx-large')
    ax.set_title("Linear regression predictions")
    plt.show()


def plot_logistic_regression(x_train, y_train):
    # Create model
    model = Sequential(
        [
            Dense(1, input_dim=1, activation='sigmoid', name='L1')
        ]
    )
    model.summary()

    # Set pre-trained weights
    set_w = np.array([[2]])
    set_b = np.array([-4.5])
    logistic_layer = model.get_layer('L1')
    logistic_layer.set_weights([set_w, set_b])

    # Predict on input data and pre-trained weights
    prediction = model.predict(x_train)

    # Plot
    pos = y_train == 1
    neg = y_train == 0

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_train[pos], y_train[pos], marker='x', s=80, c='red', label="y=1")
    ax.scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
               edgecolors='b', lw=3)
    ax.plot(x_train, prediction, color='g')

    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('Logistic regression predictions')
    ax.legend(fontsize=12)
    plt.show()


def load_logistic_regression_data():
    x_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32).reshape(-1, 1)  # 2-D Matrix

    pos = y_train == 1
    neg = y_train == 0

    fig, ax = plt.subplots(1, 1)
    ax.scatter(x_train[pos], y_train[pos], marker='x', s=80, c='red', label="y=1")
    ax.scatter(x_train[neg], y_train[neg], marker='o', s=100, label="y=0", facecolors='none',
               edgecolors='b', lw=3)

    ax.set_ylim(-0.08, 1.1)
    ax.set_ylabel('y', fontsize=12)
    ax.set_xlabel('x', fontsize=12)
    ax.set_title('Logistic regression inputs')
    ax.legend(fontsize=12)
    plt.show()

    return x_train, y_train


if __name__ == '__main__':
    x_train, y_train = load_linear_regression_data()
    plot_linear_regression(x_train, y_train)

    x_train, y_train = load_logistic_regression_data()
    plot_logistic_regression(x_train, y_train)
