from course2_advanced_learning.week3.assignment.neural_network.find_optimal import find_optimal_regularization_alpha
from course2_advanced_learning.week3.assignment.neural_network.load_data import load_nn_data
from course2_advanced_learning.week3.assignment.neural_network.models import build_nn_model_1, build_nn_model_2, \
    build_nn_model_3
from course2_advanced_learning.week3.assignment.neural_network.plot_data import plot_nn_data
from course2_advanced_learning.week3.assignment.neural_network.predict import predict_on_nn_model
from course2_advanced_learning.week3.assignment.regression.find_optimal_alpha import find_optimal_alpha
from course2_advanced_learning.week3.assignment.regression.find_optimal_degree import find_optimal_degree
from course2_advanced_learning.week3.assignment.regression.load_data import load_data, load_more_data
from course2_advanced_learning.week3.assignment.regression.plot_data import plot_data
from course2_advanced_learning.week3.assignment.regression.regression_model import build_model, evaluate_model


def test_regression():
    x_train, y_train, x_test, y_test, x_ideal, y_ideal = load_data()
    plot_data(x_ideal, y_ideal, x_train, y_train, x_test, y_test)
    model, poly, x_train_mapped_scaled, scaler_poly = build_model(x_train, y_train, 10)
    evaluate_model(model, poly, x_train_mapped_scaled, y_train, x_test, y_test, scaler_poly)
    x_train, y_train, x_cv, y_cv, x_test, y_test = load_more_data()
    optimal_degree = find_optimal_degree(x_train, y_train, x_cv, y_cv)
    optimal_alpha = find_optimal_alpha(x_train, y_train, x_cv, y_cv, optimal_degree)


def test_neural_network():
    x_train, y_train, x_cv, y_cv, x_test, y_test = load_nn_data()
    plot_nn_data(x_train, y_train)

    complex_model = build_nn_model_1(x_train, y_train)
    simple_model = build_nn_model_2(x_train, y_train)
    regularized_model = build_nn_model_3(x_train, y_train)

    predict_on_nn_model(complex_model, 'complex_model', x_train, y_train, x_cv, y_cv)
    predict_on_nn_model(simple_model, 'simple_model', x_train, y_train, x_cv, y_cv)
    predict_on_nn_model(regularized_model, 'regularized_model', x_train, y_train, x_cv, y_cv)

    find_optimal_regularization_alpha(x_train, y_train, x_cv, y_cv)

if __name__ == '__main__':
    # test_regression()
    test_neural_network()


