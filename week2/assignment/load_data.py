import numpy as np


def load_data():
    x_train = np.concatenate((np.random.uniform(low=5.0, high=9.5, size=(70,)),
                              np.random.uniform(low=9.5, high=15.5, size=(20,)),
                              np.random.uniform(low=15.5, high=22.5, size=(7,))),
                             axis=0)
    y_train = np.concatenate((np.random.uniform(low=-2.0, high=7, size=(70,)),
                              np.random.uniform(low=4, high=15, size=(20,)),
                              np.random.uniform(low=10, high=25, size=(7,))),
                             axis=0)

    return x_train, y_train
