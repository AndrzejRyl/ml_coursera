import copy, math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression


def test_scikit():
    x = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    lr_model = LogisticRegression()
    lr_model.fit(x, y)

    y_pred = lr_model.predict(x)

    print("Prediction on training set:", y_pred)

    print("Accuracy on training set:", lr_model.score(x, y))

    # Plot the linear fit
    plt.plot(x[:, 0], y_pred, c="b", label="Predictions")

    # Create a scatter plot of the data.
    plt.scatter(x[:, 0], y, marker='x', c='r', label="Actual data")

    # Set the title
    plt.title("Logistic regression with scikit")
    # Set the y-axis label
    plt.ylabel('Y')
    # Set the x-axis label
    plt.xlabel('X')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_scikit()
