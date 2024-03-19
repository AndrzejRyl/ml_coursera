import numpy as np

from course1_supervised_learning.week3.assignment.sigmoid_function import sigmoid


def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    ### START CODE HERE ###
    # Loop over each example
    for i in range(m):
        z_i = np.dot(X[i], w) + b

        # Calculate the prediction for this example
        f_wb = sigmoid(z_i)

        # Apply the threshold
        p[i] = f_wb >= 0.5
    ### END CODE HERE ###
    return p
