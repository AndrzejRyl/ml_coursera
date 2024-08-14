import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

from course2_advanced_learning.week3.assignment.regression.regression_model import build_model


def find_optimal_degree(x_train, y_train, x_cv, y_cv):
    max_degree = 9
    err_train = np.zeros(max_degree)
    err_cv = np.zeros(max_degree)

    for degree in range(max_degree):
        lmodel, poly_features, x_train_scaled, scaler_poly = build_model(x_train, y_train, degree + 1)

        lmodel.fit(x_train_scaled, y_train)
        yhat = lmodel.predict(x_train_scaled)
        err_train[degree] = mean_squared_error(y_train, yhat) / 2

        x_cv_mapped = poly_features.transform(x_cv)
        x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)
        yhat = lmodel.predict(x_cv_mapped_scaled)
        err_cv[degree] = mean_squared_error(y_cv, yhat) / 2

    optimal_degree = np.argmin(err_cv) + 1

    print("Optimal degree is %s" % optimal_degree)

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(range(max_degree), err_train, color="orange", label="Train error", lw=1)
    ax.plot(range(max_degree), err_cv, color="red", label="Cross error", lw=1)
    ax.set_title("Train, Cross errors", fontsize=14)
    ax.legend(fontsize=12)
    plt.show()

    return optimal_degree
