import numpy as np


def compute_cost_flexible(x, y, w, b):
    """
    compute cost
    Args:
      x (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters
      b (scalar)       : model parameter

    Returns:
      cost (scalar): cost
    """
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        cost += (np.dot(x[i], w) + b - y[i]) ** 2
    cost /= (2 * m)
    return cost
