from week2.assignment.compute_cost import compute_cost
from week2.assignment.compute_gradient import compute_gradient
from week2.assignment.gradient_descent import gradient_descent
from week2.assignment.load_data import load_data
from week2.assignment.plot_data import plot_data
from week2.assignment.predict import plot_linear_fit
import numpy as np


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


if __name__ == '__main__':
    x_train, y_train = load_data()
    plot_data(x_train, y_train)

    test_cost_function()
    test_gradient_function()
    run_gradient_descent()
