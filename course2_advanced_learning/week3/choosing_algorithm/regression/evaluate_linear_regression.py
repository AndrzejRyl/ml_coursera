from sklearn.metrics import mean_squared_error


def evaluate_linear_regression(linear_model, x_train_scaled, y_train, x_cv, y_cv, scaler_linear):
    # Feed the scaled training set and get the predictions
    yhat = linear_model.predict(x_train_scaled)

    # Use scikit-learn's utility function and divide by 2
    print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

    # for-loop implementation
    total_squared_error = 0

    for i in range(len(yhat)):
        squared_error_i = (yhat[i] - y_train[i]) ** 2
        total_squared_error += squared_error_i

    mse = total_squared_error / (2 * len(yhat))

    print(f"training MSE (for-loop implementation): {mse.squeeze()}")

    # Scale the cross validation set using the mean and standard deviation of the training set
    x_cv_scaled = scaler_linear.transform(x_cv)

    print(f"Mean used to scale the CV set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Standard deviation used to scale the CV set: {scaler_linear.scale_.squeeze():.2f}")

    # Feed the scaled cross validation set
    yhat = linear_model.predict(x_cv_scaled)

    # Use scikit-learn's utility function and divide by 2
    print(f"Cross validation MSE: {mean_squared_error(y_cv, yhat) / 2}")
