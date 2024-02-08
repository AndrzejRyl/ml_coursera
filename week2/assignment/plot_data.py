import matplotlib.pyplot as plt


def plot_data(x_train, y_train):
    # Plot the data points
    plt.scatter(x_train, y_train, marker='x', c='r')
    # Set the title
    plt.title("Profits vs population")
    # Set the y-axis label
    plt.ylabel('Profits (in 10.000s $)')
    # Set the x-axis label
    plt.xlabel('Population (10.000s sqft)')
    plt.show()
