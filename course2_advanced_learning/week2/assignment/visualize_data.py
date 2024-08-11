import numpy as np
import matplotlib.pyplot as plt


def visualize_data(x, y):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    m, n = x.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        x_random_reshaped = x[random_index].reshape((20, 20))

        # Display the image
        ax.imshow(x_random_reshaped, cmap='gray')

        # Display the label above the image
        ax.set_title(y[random_index, 0])
        ax.set_axis_off()

    fig.suptitle("Input data", fontsize=16)
    plt.show()
