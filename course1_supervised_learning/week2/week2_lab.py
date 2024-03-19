import numpy as np
import time


def create_vectors():
    # NumPy's routines which allocate memory and fill arrays with value
    a = np.zeros(4)
    print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.zeros((4,))
    print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.random.random_sample(4)
    print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

    # NumPy's routines which allocate memory and fill arrays with value but do not accept shape as input argument
    a = np.arange(4.)
    print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
    a = np.random.rand(4)
    print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

    # NumPy's routines which allocate memory and fill with user specified values
    a = np.array([5, 4, 3, 2])
    print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
    a = np.array([5., 4, 3, 2])
    print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


def simple_operations():
    a = np.array([1, 2, 3, 4])
    print(f"a             : {a}")
    # negate elements of a
    b = -a
    print(f"b = -a        : {b}")

    # sum all elements of a, returns a scalar
    b = np.sum(a)
    print(f"b = np.sum(a) : {b}")

    b = np.mean(a)
    print(f"b = np.mean(a): {b}")

    b = a ** 2
    print(f"b = a**2      : {b}")

    a = np.array([1, 2, 3, 4])
    b = np.array([-1, -2, 3, 4])
    print(f"Binary operators work element wise: {a + b}")


def dot_product():
    # test 1-D
    a = np.array([1, 2, 3, 4])
    b = np.array([-1, 4, 3, 2])
    c = np.dot(a, b)
    print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ")
    c = np.dot(b, a)
    print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")


def classic_dot(a, b):
    """
   Compute the dot product of two vectors

    Args:
      a (ndarray (n,)):  input vector
      b (ndarray (n,)):  input vector with same dimension as a

    Returns:
      x (scalar):
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


def why_use_numpy_dot():
    np.random.seed(1)
    a = np.random.rand(10000000)  # very large arrays
    b = np.random.rand(10000000)

    tic = time.time()  # capture start time
    c = np.dot(a, b)
    toc = time.time()  # capture end time

    print(f"np.dot(a, b) =  {c:.4f}")
    print(f"Vectorized version duration: {1000 * (toc - tic):.4f} ms ")

    tic = time.time()  # capture start time
    c = classic_dot(a, b)
    toc = time.time()  # capture end time

    print(f"my_dot(a, b) =  {c:.4f}")
    print(f"loop version duration: {1000 * (toc - tic):.4f} ms ")

    del (a)
    del (b)  # remove these big arrays from memory


def reshape_matrix():
    # vector indexing operations on matrices
    a = np.arange(6).reshape(-1, 2)  # reshape is a convenient way to create matrices
    # The -1 argument tells the routine to compute the number of rows given the size of the array
    # and the number of columns.
    print(f"a.shape: {a.shape}, \na= {a}")

    # access an element
    print(
        f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

    # access a row
    print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")


def slicing_2d():
    # vector 2-D slicing operations
    a = np.arange(20).reshape(-1, 10)
    print(f"a = \n{a}")

    # access 5 consecutive elements (start:stop:step)
    print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

    # access 5 consecutive elements (start:stop:step) in two rows
    print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

    # access all elements
    print("a[:,:] = \n", a[:, :], ",  a[:,:].shape =", a[:, :].shape)

    # access all elements in one row (very common usage)
    print("a[1,:] = ", a[1, :], ",  a[1,:].shape =", a[1, :].shape, "a 1-D array")
    # same as
    print("a[1]   = ", a[1], ",  a[1].shape   =", a[1].shape, "a 1-D array")


if __name__ == '__main__':
    # create_vectors()
    # simple_operations()
    # dot_product()
    # why_use_numpy_dot()
    reshape_matrix()
    slicing_2d()
