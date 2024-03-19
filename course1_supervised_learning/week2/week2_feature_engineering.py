import numpy as np
import matplotlib.pyplot as plt

from course1_supervised_learning.week2.week2_feature_scaling import zscore_normalize_features
from course1_supervised_learning.week2.week2_gradient_desc_multiple_vals import gradient_descent, compute_cost, compute_gradient

np.set_printoptions(precision=2)


def standard_features():
    x = np.arange(0, 20, 1)
    y = 1 + x ** 2
    X = x.reshape(-1, 1)

    # run gradient descent
    w_final, b_final, J_hist = gradient_descent(X, y, [0], 0,
                                                compute_cost, compute_gradient,
                                                1e-7, 100000)

    print(w_final)
    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.title("no feature engineering")
    plt.plot(x, X @ w_final + b_final, label="Predicted Value")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def square_engineered_features():
    # create target data
    x = np.arange(0, 20, 1)
    y = 1 + x ** 2

    # Engineer features
    X = x ** 2  # <-- added engineered feature

    X = X.reshape(-1, 1)  # X should be a 2-D Matrix
    w_final, b_final, J_hist = gradient_descent(X, y, [0], 0,
                                                compute_cost, compute_gradient,
                                                1e-7, 10000)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.title("Added x**2 feature")
    plt.plot(x, np.dot(X, w_final) + b_final, label="Predicted Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def multiple_engineered_features():
    # create target data
    x = np.arange(0, 20, 1)
    y = x ** 2

    # engineer features .
    X = np.c_[x, x ** 2, x ** 3]  # <-- added engineered feature

    w_final, b_final, J_hist = gradient_descent(X, y, [0, 0, 0], 0,
                                                compute_cost, compute_gradient,
                                                1e-7, 10000)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.title("x, x**2, x**3 features")
    plt.plot(x, X @ w_final + b_final, label="Predicted Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def multiple_engineered_features_with_scaling():
    # create target data
    x = np.arange(0, 20, 1)
    y = x ** 2
    X = np.c_[x, x ** 2, x ** 3]

    # add mean_normalization
    X, _, _ = zscore_normalize_features(X)

    w_final, b_final, J_hist = gradient_descent(X, y, [0, 0, 0], 0,
                                                compute_cost, compute_gradient,
                                                1e-1, 100000)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.title("Normalized x, x**2, x**3 features")
    plt.plot(x, X @ w_final + b_final, label="Predicted Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


def complex_function():
    x = np.arange(0, 20, 1)
    y = np.cos(x / 2)

    X = np.c_[x, x ** 2, x ** 3, x ** 4, x ** 5, x ** 6, x ** 7, x ** 8, x ** 9, x ** 10, x ** 11, x ** 12, x ** 13]
    X, _, _ = zscore_normalize_features(X)

    w_final, b_final, J_hist = gradient_descent(X, y, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0,
                                                compute_cost, compute_gradient,
                                                1e-1, 1000000)

    plt.scatter(x, y, marker='x', c='r', label="Actual Value")
    plt.title("Normalized x x**2, x**3 feature")
    plt.plot(x, X @ w_final + b_final, label="Predicted Value")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Fitting straight line
    standard_features()

    # Fitting ideal function but we basically would have to know that x**2 matches it
    square_engineered_features()

    # Guessing higher rank function but it requires lots of fitting
    multiple_engineered_features()

    # Guessing higher rank function but with scaled features allows us to converge faster
    multiple_engineered_features_with_scaling()

    # Super complex function with engineered features
    complex_function()
