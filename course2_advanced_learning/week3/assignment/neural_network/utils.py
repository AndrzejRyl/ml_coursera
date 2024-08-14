def eval_category_error_nn(y, yhat):
    """
    Calculate the categorization error
    Args:
      y    : (ndarray  Shape (m,) or (m,1))  target value of each example
      yhat : (ndarray  Shape (m,) or (m,1))  predicted value of each example
    Returns:|
      cerr: (scalar)
    """
    m = len(y)
    category_error = 0
    for i in range(m):
        if y[i] != yhat[i]:
            category_error += 1 / m

    return category_error
