
import matplotlib.pyplot as plt


def plot_data(x_ideal, y_ideal, x_train, y_train, x_test, y_test):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot(x_ideal, y_ideal, "--", color="orangered", label="y_ideal", lw=1)
    ax.set_title("Training, Test", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.scatter(x_train, y_train, color="red", label="train")
    ax.scatter(x_test, y_test, color="blue", label="test")
    ax.legend(loc='upper left')
    plt.show()
