import matplotlib.pyplot as plt


def plot_fit_flexible(x, y_train, w, b, features):
    m, n = x.shape  # (number of examples, number of features)

    # Plot the linear fit
    plt.plot(features, x @ w + b, c="b")

    # Create a scatter plot of the data.
    plt.scatter(features, y_train, marker='x', c='r')

    # Set the title
    plt.title("Profits vs. Population per city. Flexible. Rank %s" % n)
    # Set the y-axis label
    plt.ylabel('Profit in $10,000')
    # Set the x-axis label
    plt.xlabel('Population of City in 10,000s')
    plt.show()
