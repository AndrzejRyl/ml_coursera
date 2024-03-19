import numpy as np

from course1_supervised_learning.week2.assignment.additional.compute_cost_flexible import compute_cost_flexible
from course1_supervised_learning.week2.assignment.additional.compute_gradient_flexible import compute_gradient_flexible
from course1_supervised_learning.week2.assignment.additional.gradient_descent_flexible import gradient_descent_flexible
from course1_supervised_learning.week2.assignment.additional.plot_fit_flexible import plot_fit_flexible
from course1_supervised_learning.week2.assignment.compute_cost import compute_cost
from course1_supervised_learning.week2.assignment.compute_gradient import compute_gradient
from course1_supervised_learning.week2.assignment.gradient_descent import gradient_descent
from course1_supervised_learning.week2.assignment.load_data import load_data
from course1_supervised_learning.week2.assignment.plot_data import plot_data
from course1_supervised_learning.week2.assignment.plot_linear_fit import plot_linear_fit
from course1_supervised_learning.week2.week2_feature_scaling import zscore_normalize_features

import matplotlib.pyplot as plt


def test_cost_function():
    # Compute cost with some initial values for paramaters w, b
    initial_w = 2
    initial_b = 1
    cost = compute_cost(x_train, y_train, initial_w, initial_b)
    print(type(cost))
    print(f'Cost at initial w: {cost:.3f}')


def test_gradient_function():
    # Compute and display gradient with w initialized to zeroes
    initial_w = 0
    initial_b = 0
    tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
    print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)


def run_gradient_descent():
    # initialize fitting parameters. Recall that the shape of w is (n,)
    initial_w = 0.
    initial_b = 0.
    # some gradient descent settings
    iterations = 1500
    alpha = 0.01
    w, b, _, _ = gradient_descent(x_train, y_train, initial_w, initial_b,
                                  compute_cost, compute_gradient, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    plot_linear_fit(x_train, y_train, w, b)


def run_rank1_gradient_descent():
    x_rank1 = np.c_[x_train]
    # initialize fitting parameters. Recall that the shape of w is (n,)
    initial_w = [0.]
    initial_b = 0.
    # some gradient descent settings
    iterations = 1500
    alpha = 0.01
    w, b, j_hist = gradient_descent_flexible(x_rank1, y_train, initial_w, initial_b,
                                             compute_cost_flexible, compute_gradient_flexible, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    plot_fit_flexible(x_rank1, y_train, w, b, x_train)

    return x_rank1, w, b, j_hist


def run_rank2_gradient_descent():
    x_rank2 = np.c_[x_train, x_train ** 2]

    x_rank2, _, _ = zscore_normalize_features(x_rank2)
    # initialize fitting parameters. Recall that the shape of w is (n,)
    initial_w = [0., 0.]
    initial_b = 0.
    # some gradient descent settings
    iterations = 1500
    alpha = 0.01
    w, b, j_hist = gradient_descent_flexible(x_rank2, y_train, initial_w, initial_b,
                                             compute_cost_flexible, compute_gradient_flexible, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    plot_fit_flexible(x_rank2, y_train, w, b, x_train)

    return x_rank2, w, b, j_hist


def run_rank3_gradient_descent():
    x_rank3 = np.c_[x_train, x_train ** 2, x_train ** 3]

    x_rank3, _, _ = zscore_normalize_features(x_rank3)
    # initialize fitting parameters. Recall that the shape of w is (n,)
    initial_w = [0., 0., 0.]
    initial_b = 0.
    # some gradient descent settings
    iterations = 1500
    alpha = 0.01
    w, b, j_hist = gradient_descent_flexible(x_rank3, y_train, initial_w, initial_b,
                                             compute_cost_flexible, compute_gradient_flexible, alpha, iterations)
    print("w,b found by gradient descent:", w, b)

    plot_fit_flexible(x_rank3, y_train, w, b, x_train)

    return x_rank3, w, b, j_hist


def plot_results_for_multiple_ranks():
    fig, ((pred1, pred2, pred3), (jhist1, jhist2, jhist3)) = plt.subplots(2, 3, constrained_layout=True, figsize=(12, 8))

    pred1.plot(x_train, x_rank1 @ w_rank1 + b_rank1, c="b")
    pred1.scatter(x_train, y_train, marker='x', c='r')
    pred2.plot(x_train, x_rank2 @ w_rank2 + b_rank2, c="b")
    pred2.scatter(x_train, y_train, marker='x', c='r')
    pred3.plot(x_train, x_rank3 @ w_rank3 + b_rank3, c="b")
    pred3.scatter(x_train, y_train, marker='x', c='r')

    jhist1.plot(j_hist_rank1)
    jhist2.plot(j_hist_rank2)
    jhist3.plot(j_hist_rank3)

    pred1.set_title("Predictions rank1")
    pred2.set_title("Predictions rank2")
    pred3.set_title("Predictions rank3")
    pred1.set_ylabel('Profit in $10,000')
    pred2.set_ylabel('Profit in $10,000')
    pred3.set_ylabel('Profit in $10,000')
    pred1.set_xlabel('Population of City in 10,000s')
    pred2.set_xlabel('Population of City in 10,000s')
    pred3.set_xlabel('Population of City in 10,000s')

    jhist1.set_title("Cost rank1")
    jhist2.set_title("Cost rank2")
    jhist3.set_title("Cost rank3")
    jhist1.set_ylabel('Cost')
    jhist2.set_ylabel('Cost')
    jhist3.set_ylabel('Cost')
    jhist1.set_xlabel('iteration step')
    jhist2.set_xlabel('iteration step')
    jhist3.set_xlabel('iteration step')

    plt.show()


if __name__ == '__main__':
    x_train, y_train = load_data()
    plot_data(x_train, y_train)

    test_cost_function()
    test_gradient_function()
    run_gradient_descent()

    # Additional - make more flexible approach and experiment with higher rank functions
    x_rank1, w_rank1, b_rank1, j_hist_rank1 = run_rank1_gradient_descent()
    x_rank2, w_rank2, b_rank2, j_hist_rank2 = run_rank2_gradient_descent()
    x_rank3, w_rank3, b_rank3, j_hist_rank3 = run_rank3_gradient_descent()

    plot_results_for_multiple_ranks()
