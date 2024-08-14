from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def train_polynomial_regression(x_train, y_train, degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)

    # Compute the number of features and transform the training set
    x_train_mapped = poly.fit_transform(x_train)

    # Instantiate the class
    scaler_poly = StandardScaler()

    # Compute the mean and standard deviation of the training set then transform it
    x_train_mapped_scaled = scaler_poly.fit_transform(x_train_mapped)

    # Initialize the class
    model = LinearRegression()

    # Train the model
    model.fit(x_train_mapped_scaled, y_train)

    return model, poly, x_train_mapped_scaled, scaler_poly
