import tensorflow as tf
import matplotlib.pyplot as plt
from course2_advanced_learning.week3.assignment.neural_network.models import build_nn_model_3
from course2_advanced_learning.week3.assignment.neural_network.predict import predict_on_nn_model


def find_optimal_regularization_alpha(x_train, y_train, x_cv, y_cv):
    tf.random.set_seed(1234)
    lambdas = [0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3]
    models = [None] * len(lambdas)

    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        models[i] = build_nn_model_3(x_train, y_train, regularization_alpha=lambda_)

    training_set_errors = []
    cross_set_errors = []

    for i in range(len(lambdas)):
        lambda_ = lambdas[i]
        model = models[i]

        training_set_error, cross_set_error = predict_on_nn_model(
            model=model,
            model_name=f"Regularized with alpha {lambda_}",
            x_train=x_train,
            y_train=y_train,
            x_cv=x_cv,
            y_cv=y_cv,
        )
        training_set_errors.append(training_set_error)
        cross_set_errors.append(cross_set_error)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(lambdas, training_set_errors, color="r", label="Training set errors", lw=1)
    ax.plot(lambdas, cross_set_errors, color="b", label="Cross set errors", lw=1)
    ax.set_title("Training, Cross errors", fontsize=14)
    ax.set_xlabel("Lambda")
    ax.set_ylabel("Errors")
    ax.legend(loc='upper left')
    plt.show()

    optimal_alpha_idx = cross_set_errors.index(min(cross_set_errors))
    print(f"Optimal alpha is {lambdas[optimal_alpha_idx]}")
