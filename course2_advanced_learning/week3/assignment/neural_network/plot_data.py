from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

COLORS = ['r', 'b', 'y', 'g', 'k', 'c']


def plot_nn_data(x_train, y_train):
    data = defaultdict(list)

    for i, y in enumerate(y_train):
        data[y[0]].append(x_train[i])

    for class_idx, values in data.items():
        plt.scatter(
            np.array(values)[:, 0],
            np.array(values)[:, 1],
            marker='x',
            c=COLORS[class_idx],
            label='Class %s' % class_idx
        )

    plt.legend(loc="upper right")
    plt.show()
