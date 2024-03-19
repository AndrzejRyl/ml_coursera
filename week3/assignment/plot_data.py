import matplotlib.pyplot as plt
import numpy as np


def plot_data(x_train, y_train):
    admitted_x_train = []
    not_admitted_x_train = []

    for i, y in enumerate(y_train):
        if y == 1:
            admitted_x_train.append(x_train[i])
        else:
            not_admitted_x_train.append(x_train[i])

    plt.scatter(np.array(admitted_x_train)[:, 0], np.array(admitted_x_train)[:, 1], marker='x', c='r', label='Admitted')
    plt.scatter(np.array(not_admitted_x_train)[:, 0], np.array(not_admitted_x_train)[:, 1], marker='o', c='y',
                label='Not admitted')

    # Set the y-axis label
    plt.ylabel('Exam 2 score')
    # Set the x-axis label
    plt.xlabel('Exam 1 score')
    plt.legend(loc="upper right")
    plt.show()
