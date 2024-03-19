import matplotlib.pyplot as plt
import numpy as np

from course1_supervised_learning.week3.week3_sigmoid_function import sigmoid


def load_data():
    x_train = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  # (m,n)
    y_train = np.array([0, 0, 0, 1, 1, 1])

    return x_train, y_train


def plot_data(x_train, y_train):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    x_0 = [x[0] for i, x in enumerate(x_train) if y_train[i] == 0]
    x_1 = [x[1] for i, x in enumerate(x_train) if y_train[i] == 0]
    plt.scatter(x_0, x_1, marker='o', c='b', label='y=0')

    x_0 = [x[0] for i, x in enumerate(x_train) if y_train[i] == 1]
    x_1 = [x[1] for i, x in enumerate(x_train) if y_train[i] == 1]
    plt.scatter(x_0, x_1, marker='x', c='r', label='y=1')

    # Set both axes to be from 0-4
    ax.axis([0, 4, 0, 3.5])
    ax.set_ylabel('$x_1$', fontsize=12)
    ax.set_xlabel('$x_0$', fontsize=12)
    plt.legend()
    plt.show()


def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)

    cost = cost / m
    return cost


def test_compute_cost():
    w_array1 = np.array([1, 1])
    b_1 = -3
    w_array2 = np.array([1, 1])
    b_2 = -4

    print("Cost for b = -3 : ", compute_cost_logistic(x_train, y_train, w_array1, b_1))
    print("Cost for b = -4 : ", compute_cost_logistic(x_train, y_train, w_array2, b_2))


if __name__ == '__main__':
    x_train, y_train = load_data()
    plot_data(x_train, y_train)
    test_compute_cost()
