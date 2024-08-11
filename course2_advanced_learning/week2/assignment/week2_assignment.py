# In this exercise, you will use a neural network to recognize all handwritten digits.
# This is a multi-class classification task. Automated handwritten digit recognition is widely used today -
# from recognizing zip codes (postal codes) on mail envelopes to recognizing amounts written on bank checks.
#
# This exercise will show you how the methods you have learned can be used for this classification task.
import tensorflow as tf
from tensorflow.keras.layers import Dense, Normalization, Input
from tensorflow.keras.models import Sequential

from course2_advanced_learning.week2.assignment.load_data import load_data
from course2_advanced_learning.week2.assignment.visualize_data import visualize_data
from course2_advanced_learning.week2.assignment.visualize_predictions import visualize_predictions


def build_model():
    model = Sequential(
        [
            Input(shape=(400,)),  # specify input size
            Dense(25, activation='relu', name='layer1'),
            Dense(15, activation='relu', name='layer2'),
            Dense(10, activation='linear', name='layer3'),
        ], name="my_model"
    )

    return model


def train_model(model, x, y):
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        x, y,
        epochs=20
    )


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_data()
    visualize_data(x_train, y_train)
    model = build_model()
    model.summary()
    train_model(model, x_train, y_train)
    visualize_predictions(x_test, y_test, model)
