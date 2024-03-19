import matplotlib.pyplot as plt
import numpy as np

from course1_supervised_learning.week2.week2_gradient_desc_multiple_vals import gradient_descent, compute_cost, compute_gradient

np.set_printoptions(precision=2)


def load_dataset():
    X_train = [[1.24e+03, 3.00e+00, 1.00e+00, 6.40e+01]
        , [1.95e+03, 3.00e+00, 2.00e+00, 1.70e+01]
        , [1.72e+03, 3.00e+00, 2.00e+00, 4.20e+01]
        , [1.96e+03, 3.00e+00, 2.00e+00, 1.50e+01]
        , [1.31e+03, 2.00e+00, 1.00e+00, 1.40e+01]
        , [8.64e+02, 2.00e+00, 1.00e+00, 6.60e+01]
        , [1.84e+03, 3.00e+00, 1.00e+00, 1.70e+01]
        , [1.03e+03, 3.00e+00, 1.00e+00, 4.30e+01]
        , [3.19e+03, 4.00e+00, 2.00e+00, 8.70e+01]
        , [7.88e+02, 2.00e+00, 1.00e+00, 8.00e+01]
        , [1.20e+03, 2.00e+00, 2.00e+00, 1.70e+01]
        , [1.56e+03, 2.00e+00, 1.00e+00, 1.80e+01]
        , [1.43e+03, 3.00e+00, 1.00e+00, 2.00e+01]
        , [1.22e+03, 2.00e+00, 1.00e+00, 1.50e+01]
        , [1.09e+03, 2.00e+00, 1.00e+00, 6.40e+01]
        , [8.48e+02, 1.00e+00, 1.00e+00, 1.70e+01]
        , [1.68e+03, 3.00e+00, 2.00e+00, 2.30e+01]
        , [1.77e+03, 3.00e+00, 2.00e+00, 1.80e+01]
        , [1.04e+03, 3.00e+00, 1.00e+00, 4.40e+01]
        , [1.65e+03, 2.00e+00, 1.00e+00, 2.10e+01]
        , [1.09e+03, 2.00e+00, 1.00e+00, 3.50e+01]
        , [1.32e+03, 3.00e+00, 1.00e+00, 1.40e+01]
        , [1.59e+03, 0.00e+00, 1.00e+00, 2.00e+01]
        , [9.72e+02, 2.00e+00, 1.00e+00, 7.30e+01]
        , [1.10e+03, 3.00e+00, 1.00e+00, 3.70e+01]
        , [1.00e+03, 2.00e+00, 1.00e+00, 5.10e+01]
        , [9.04e+02, 3.00e+00, 1.00e+00, 5.50e+01]
        , [1.69e+03, 3.00e+00, 1.00e+00, 1.30e+01]
        , [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02]
        , [1.42e+03, 3.00e+00, 2.00e+00, 1.90e+01]
        , [1.16e+03, 3.00e+00, 1.00e+00, 5.20e+01]
        , [1.94e+03, 3.00e+00, 2.00e+00, 1.20e+01]
        , [1.22e+03, 2.00e+00, 2.00e+00, 7.40e+01]
        , [2.48e+03, 4.00e+00, 2.00e+00, 1.60e+01]
        , [1.20e+03, 2.00e+00, 1.00e+00, 1.80e+01]
        , [1.84e+03, 3.00e+00, 2.00e+00, 2.00e+01]
        , [1.85e+03, 3.00e+00, 2.00e+00, 5.70e+01]
        , [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01]
        , [1.10e+03, 2.00e+00, 2.00e+00, 9.70e+01]
        , [1.78e+03, 3.00e+00, 2.00e+00, 2.80e+01]
        , [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01]
        , [1.78e+03, 4.00e+00, 2.00e+00, 1.07e+02]
        , [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02]
        , [1.55e+03, 3.00e+00, 1.00e+00, 1.60e+01]
        , [1.95e+03, 3.00e+00, 2.00e+00, 1.60e+01]
        , [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01]
        , [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01]
        , [8.16e+02, 2.00e+00, 1.00e+00, 5.80e+01]
        , [1.35e+03, 3.00e+00, 1.00e+00, 2.10e+01]
        , [1.57e+03, 3.00e+00, 1.00e+00, 1.40e+01]
        , [1.49e+03, 3.00e+00, 1.00e+00, 5.70e+01]
        , [1.51e+03, 2.00e+00, 1.00e+00, 1.60e+01]
        , [1.10e+03, 3.00e+00, 1.00e+00, 2.70e+01]
        , [1.76e+03, 3.00e+00, 2.00e+00, 2.40e+01]
        , [1.21e+03, 2.00e+00, 1.00e+00, 1.40e+01]
        , [1.47e+03, 3.00e+00, 2.00e+00, 2.40e+01]
        , [1.77e+03, 3.00e+00, 2.00e+00, 8.40e+01]
        , [1.65e+03, 3.00e+00, 1.00e+00, 1.90e+01]
        , [1.03e+03, 3.00e+00, 1.00e+00, 6.00e+01]
        , [1.12e+03, 2.00e+00, 2.00e+00, 1.60e+01]
        , [1.15e+03, 3.00e+00, 1.00e+00, 6.20e+01]
        , [8.16e+02, 2.00e+00, 1.00e+00, 3.90e+01]
        , [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01]
        , [1.39e+03, 3.00e+00, 1.00e+00, 6.40e+01]
        , [1.60e+03, 3.00e+00, 2.00e+00, 2.90e+01]
        , [1.22e+03, 3.00e+00, 1.00e+00, 6.30e+01]
        , [1.07e+03, 2.00e+00, 1.00e+00, 1.00e+02]
        , [2.60e+03, 4.00e+00, 2.00e+00, 2.20e+01]
        , [1.43e+03, 3.00e+00, 1.00e+00, 5.90e+01]
        , [2.09e+03, 3.00e+00, 2.00e+00, 2.60e+01]
        , [1.79e+03, 4.00e+00, 2.00e+00, 4.90e+01]
        , [1.48e+03, 3.00e+00, 2.00e+00, 1.60e+01]
        , [1.04e+03, 3.00e+00, 1.00e+00, 2.50e+01]
        , [1.43e+03, 3.00e+00, 1.00e+00, 2.20e+01]
        , [1.16e+03, 3.00e+00, 1.00e+00, 5.30e+01]
        , [1.55e+03, 3.00e+00, 2.00e+00, 1.20e+01]
        , [1.98e+03, 3.00e+00, 2.00e+00, 2.20e+01]
        , [1.06e+03, 3.00e+00, 1.00e+00, 5.30e+01]
        , [1.18e+03, 2.00e+00, 1.00e+00, 9.90e+01]
        , [1.36e+03, 2.00e+00, 1.00e+00, 1.70e+01]
        , [9.60e+02, 3.00e+00, 1.00e+00, 5.10e+01]
        , [1.46e+03, 3.00e+00, 2.00e+00, 1.60e+01]
        , [1.45e+03, 3.00e+00, 2.00e+00, 2.50e+01]
        , [1.21e+03, 2.00e+00, 1.00e+00, 1.50e+01]
        , [1.55e+03, 3.00e+00, 2.00e+00, 1.60e+01]
        , [8.82e+02, 3.00e+00, 1.00e+00, 4.90e+01]
        , [2.03e+03, 4.00e+00, 2.00e+00, 4.50e+01]
        , [1.04e+03, 3.00e+00, 1.00e+00, 6.20e+01]
        , [1.62e+03, 3.00e+00, 1.00e+00, 1.60e+01]
        , [8.03e+02, 2.00e+00, 1.00e+00, 8.00e+01]
        , [1.43e+03, 3.00e+00, 2.00e+00, 2.10e+01]
        , [1.66e+03, 3.00e+00, 1.00e+00, 6.10e+01]
        , [1.54e+03, 3.00e+00, 1.00e+00, 1.60e+01]
        , [9.48e+02, 3.00e+00, 1.00e+00, 5.30e+01]
        , [1.22e+03, 2.00e+00, 2.00e+00, 1.20e+01]
        , [1.43e+03, 2.00e+00, 1.00e+00, 4.30e+01]
        , [1.66e+03, 3.00e+00, 2.00e+00, 1.90e+01]
        , [1.21e+03, 3.00e+00, 1.00e+00, 2.00e+01]
        , [1.05e+03, 2.00e+00, 1.00e+00, 6.50e+01]]

    y_train = [300., 509.8, 394., 540., 415., 230., 560., 294., 718.2, 200.,
               302., 468., 374.2, 388., 282., 311.8, 401., 449.8, 301., 502.,
               340., 400.28, 572., 264., 304., 298., 219.8, 490.7, 216.96, 368.2,
               280., 526.87, 237., 562.43, 369.8, 460., 374., 390., 158., 426.,
               390., 277.77, 216.96, 425.8, 504., 329., 464., 220., 358., 478.,
               334., 426.98, 290., 463., 390.8, 354., 350., 460., 237., 288.3,
               282., 249., 304., 332., 351.8, 310., 216.96, 666.34, 330., 480.,
               330.3, 348., 304., 384., 316., 430.4, 450., 284., 275., 414.,
               258., 378., 350., 412., 373., 225., 390., 267.4, 464., 174.,
               340., 430., 440., 216., 329., 388., 390., 356., 257.8]

    X_features = ['size(sqft)', 'bedrooms', 'floors', 'age']

    return np.array(X_train), np.array(y_train), np.array(X_features)


def plot_dataset():
    fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for i in range(len(ax)):
        ax[i].scatter(X_train[:, i], y_train)
        ax[i].set_xlabel(X_features[i])
    ax[0].set_ylabel("Price (1000's)")
    plt.show()


def gradient_descent_with_multiple_features(X_train, y_train, iterations, alpha):
    b_init = 785.1811367994083
    w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

    # initialize parameters
    initial_w = np.zeros_like(w_init)
    initial_b = 0.
    # run gradient descent
    w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                compute_cost, compute_gradient,
                                                alpha, iterations)
    print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
    m, _ = X_train.shape
    for i in range(m):
        print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(J_hist)
    ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
    ax1.set_title("Cost vs. iteration")
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')
    ax2.set_xlabel('iteration step')
    plt.show()


def test_different_alpha(X_train, y_train):
    gradient_descent_with_multiple_features(X_train, y_train, 10, alpha=9.9e-7)
    gradient_descent_with_multiple_features(X_train, y_train, 10, alpha=9e-7)
    gradient_descent_with_multiple_features(X_train, y_train, 10, alpha=1e-7)


def zscore_normalize_features(X):
    """
    computes  X, zcore normalized by column

    Args:
      X (ndarray (m,n))     : input data, m examples, n features

    Returns:
      X_norm (ndarray (m,n)): input normalized by column
      mu (ndarray (n,))     : mean of each feature
      sigma (ndarray (n,))  : standard deviation of each feature
    """
    # find the mean of each column/feature
    mu = np.mean(X, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(X, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma

    return (X_norm, mu, sigma)


def normalize_features(X_train, y_train):
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    X_mean = (X_train - mu)
    X_norm = (X_train - mu) / sigma

    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    ax[0].scatter(X_train[:, 0], X_train[:, 3])
    ax[0].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[3])
    ax[0].set_title("unnormalized")
    ax[0].axis('equal')

    ax[1].scatter(X_mean[:, 0], X_mean[:, 3])
    ax[1].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[3])
    ax[1].set_title(r"X - $\mu$")
    ax[1].axis('equal')

    ax[2].scatter(X_norm[:, 0], X_norm[:, 3])
    ax[2].set_xlabel(X_features[0])
    ax[0].set_ylabel(X_features[3])
    ax[2].set_title(r"Z-score normalized")
    ax[2].axis('equal')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle("distribution of features before, during, after normalization")
    plt.show()


def run_on_normalized_features(X_train, y_train):
    X_norm, X_mu, X_sigma = zscore_normalize_features(X_train)
    print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
    print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train, axis=0)}")
    print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm, axis=0)}")

    gradient_descent_with_multiple_features(X_norm, y_train, 10, 1.0e-1)


if __name__ == '__main__':
    (X_train, y_train, X_features) = load_dataset()
    # plot_dataset()
    test_different_alpha(X_train, y_train)
    # normalize_features(X_train, y_train)
    run_on_normalized_features(X_train, y_train)
