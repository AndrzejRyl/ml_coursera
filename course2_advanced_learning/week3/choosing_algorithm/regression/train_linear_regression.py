from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def train_linear_regression(x_train, y_train):
    # Initialize the class
    scaler_linear = StandardScaler()

    # Compute the mean and standard deviation of the training set then transform it
    x_train_scaled = scaler_linear.fit_transform(x_train)

    print(f"Computed mean of the training set: {scaler_linear.mean_.squeeze():.2f}")
    print(f"Computed standard deviation of the training set: {scaler_linear.scale_.squeeze():.2f}")

    # Initialize the class
    linear_model = LinearRegression()

    # Train the model
    linear_model.fit(x_train_scaled, y_train)

    return linear_model, x_train_scaled, scaler_linear
