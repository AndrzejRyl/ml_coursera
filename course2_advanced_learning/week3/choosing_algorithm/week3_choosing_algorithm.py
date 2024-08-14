import numpy as np
from sklearn.model_selection import train_test_split

from course2_advanced_learning.week3.choosing_algorithm.regression.evaluate_linear_regression import \
    evaluate_linear_regression
from course2_advanced_learning.week3.choosing_algorithm.regression.evaluate_poly_regression import \
    evaluate_polynomial_regression
from course2_advanced_learning.week3.choosing_algorithm.regression.load_data import load_regression_model_data
from course2_advanced_learning.week3.choosing_algorithm.regression.plot_data import plot_regression_data
from course2_advanced_learning.week3.choosing_algorithm.regression.train_linear_regression import \
    train_linear_regression
from course2_advanced_learning.week3.choosing_algorithm.regression.train_polynomial_regression import \
    train_polynomial_regression


def choose_best_regression_model():
    x, y = load_regression_model_data()
    plot_regression_data(x, y)

    # Get 60% of the dataset as the training set. Put the remaining 40% in temporary variables: x_ and y_.
    x_train, x_, y_train, y_ = train_test_split(x, y, test_size=0.40, random_state=1)

    # Split the 40% subset above into two: one half for cross validation and the other for the test set
    x_cv, x_test, y_cv, y_test = train_test_split(x_, y_, test_size=0.50, random_state=1)

    # Delete temporary variables
    del x_, y_

    linear_model, x_train_scaled, scaler_linear = train_linear_regression(x_train, y_train)
    print("\n\nLinear regression results")
    evaluate_linear_regression(linear_model, x_train_scaled, y_train, x_cv, y_cv, scaler_linear)

    cv_errors = []
    for degree in range(1, 11):
        poly_model, poly, x_train_mapped_scaled, scaler_poly = (
            train_polynomial_regression(x_train, y_train, degree)
        )
        print("\n\nPolynomial regression of degree %s results" % degree)
        cv_error = evaluate_polynomial_regression(poly_model, poly, x_train_mapped_scaled, y_train, x_cv, y_cv,
                                                  scaler_poly)
        cv_errors.append(cv_error)

    print(f"\n\nDegree with lowest CV error is {np.argmin(cv_errors) + 1}")


def choose_best_nn_model():
    pass


if __name__ == '__main__':
    choose_best_regression_model()
    choose_best_nn_model()
