import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from course2_advanced_learning.week3.assignment.regression.regression_model import build_model


def find_optimal_alpha(x_train, y_train, x_cv, y_cv, optimal_degree):
    lambda_range = np.array([0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100])
    num_steps = len(lambda_range)
    err_train = np.zeros(num_steps)
    err_cv = np.zeros(num_steps)

    for i in range(num_steps):
        lambda_ = lambda_range[i]
        lmodel, poly_features, x_train_scaled, scaler_poly = build_model(x_train, y_train, optimal_degree, alpha=lambda_)

        lmodel.fit(x_train_scaled, y_train)
        yhat = lmodel.predict(x_train_scaled)
        err_train[i] = mean_squared_error(y_train, yhat) / 2

        x_cv_mapped = poly_features.transform(x_cv)
        x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
        yhat = lmodel.predict(x_cv_mapped_scaled)
        err_cv[i] = mean_squared_error(y_cv, yhat) / 2

    optimal_reg_idx = np.argmin(err_cv)

    print("Optimal lambda is %s" % lambda_range[optimal_reg_idx])

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(range(num_steps), err_train, color="orange", label="Train error", lw=1)
    ax.plot(range(num_steps), err_cv, color="red", label="Cross error", lw=1)
    ax.set_title("Train, Cross errors", fontsize=14)
    ax.legend(fontsize=12)
    plt.show()

    return lambda_range[optimal_reg_idx]
