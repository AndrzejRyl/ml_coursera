from sklearn.metrics import mean_squared_error


def evaluate_polynomial_regression(poly_model, poly_features, x_train_scaled, y_train, x_cv, y_cv, scaler_poly):
    # Feed the scaled training set and get the predictions
    yhat = poly_model.predict(x_train_scaled)
    print(f"Training MSE: {mean_squared_error(y_train, yhat) / 2}")

    # Add the polynomial features to the cross validation set
    x_cv_mapped = poly_features.transform(x_cv)

    # Scale the cross validation set using the mean and standard deviation of the training set
    x_cv_mapped_scaled = scaler_poly.transform(x_cv_mapped)

    # Compute the cross validation MSE
    yhat = poly_model.predict(x_cv_mapped_scaled)
    cv_error = mean_squared_error(y_cv, yhat) / 2
    print(f"Cross validation MSE: {cv_error}")

    return cv_error
