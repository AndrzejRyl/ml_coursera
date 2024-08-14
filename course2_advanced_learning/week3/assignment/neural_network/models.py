import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def build_nn_model_1(x_train, y_train):
    tf.random.set_seed(1234)

    model = Sequential(
        [
            Dense(units=120, activation='relu'),
            Dense(units=40, activation='relu'),
            Dense(units=6, activation='linear'),
        ], name="Complex"
    )
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(0.01),
    )

    model.fit(x_train, y_train, epochs=1000)
    model.summary()
    return model


def build_nn_model_2(x_train, y_train):
    tf.random.set_seed(1234)

    model = Sequential(
        [
            Dense(units=6, activation='relu'),
            Dense(units=6, activation='linear'),
        ], name="Simple"
    )
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(0.01),
    )

    model.fit(x_train, y_train, epochs=1000)
    model.summary()
    return model


def build_nn_model_3(x_train, y_train, regularization_alpha=0.1):
    tf.random.set_seed(1234)

    model = Sequential(
        [
            Dense(units=120, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_alpha)),
            Dense(units=40, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(regularization_alpha)),
            Dense(units=6, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(regularization_alpha)),
        ], name="Regularized"
    )
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=Adam(0.01),
    )

    model.fit(x_train, y_train, epochs=1000)
    model.summary()
    return model
