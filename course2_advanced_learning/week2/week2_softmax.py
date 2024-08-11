import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


def my_softmax(z):
    ez = np.exp(z)  # element-wise exponenial
    sm = ez / np.sum(ez)
    return sm


def build_classic_model(x_train, y_train):
    model = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(15, activation='relu'),
            Dense(4, activation='softmax')  # < softmax activation here
        ]
    )
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        x_train, y_train,
        epochs=10
    )

    p_nonpreferred = model.predict(x_train)
    print(p_nonpreferred[:2])
    print("largest value", np.max(p_nonpreferred), "smallest value", np.min(p_nonpreferred))

    return model


def build_numerically_adequate_model(x_train, y_train):
    preferred_model = Sequential(
        [
            Dense(25, activation='relu'),
            Dense(15, activation='relu'),
            Dense(4, activation='linear')  # <-- Note
        ]
    )
    preferred_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # <-- Note
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    preferred_model.fit(
        x_train, y_train,
        epochs=10
    )

    p_preferred = preferred_model.predict(x_train)
    print(f"two example output vectors:\n {p_preferred[:2]}")
    print("largest value", np.max(p_preferred), "smallest value", np.min(p_preferred))

    # To get probabilities in this case, you have to run it through softmax
    sm_preferred = tf.nn.softmax(p_preferred).numpy()
    print(f"two example output vectors:\n {sm_preferred[:2]}")
    print("largest value", np.max(sm_preferred), "smallest value", np.min(sm_preferred))

    return preferred_model


def load_data():
    pass


if __name__ == '__main__':
    # No point in running this. I added this to show how to build model using "from_logits"
    x_train, y_train = load_data()
    build_classic_model(x_train, y_train)
    build_numerically_adequate_model(x_train, y_train)
