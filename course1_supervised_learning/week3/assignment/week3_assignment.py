import numpy as np

from course1_supervised_learning.week3.assignment.compute_gradient import compute_gradient
from course1_supervised_learning.week3.assignment.cost_function import compute_cost
from course1_supervised_learning.week3.assignment.gradient_descent import gradient_descent
from course1_supervised_learning.week3.assignment.load_data import load_data
from course1_supervised_learning.week3.assignment.plot_data import plot_data
from course1_supervised_learning.week3.assignment.regularized.compute_gradient import compute_gradient_reg
from course1_supervised_learning.week3.assignment.regularized.cost_function import compute_cost_logistic_reg

if __name__ == '__main__':
    x_train, y_train = load_data()
    plot_data(x_train, y_train)

    np.random.seed(1)
    initial_w = 0.01 * (np.random.rand(2) - 0.5)
    initial_b = -8

    # Some gradient descent settings
    iterations = 3000
    alpha = 0.001

    w, b, J_history, _ = gradient_descent(x_train, y_train, initial_w, initial_b,
                                          compute_cost, compute_gradient, alpha, iterations, 0)

    print(w)
    print(b)
    print(J_history)

    # For more complex examples that don't have decision boundary in the form of a straight line
    w_regularization, b_regularization, J_history_regularization, _ = gradient_descent(
        x_train, y_train, initial_w, initial_b,
        compute_cost_logistic_reg, compute_gradient_reg, alpha, iterations, 0)

    print(w_regularization)
    print(b_regularization)
    print(J_history_regularization)
