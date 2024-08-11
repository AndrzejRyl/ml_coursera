import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def visualize_predictions(x, y, model):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # You do not need to modify anything in this cell

    m, n = x.shape

    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])  # [left, bottom, right, top]

    for i, ax in enumerate(axes.flat):
        # Select random indices
        random_index = np.random.randint(m)

        # Select rows corresponding to the random indices and
        # reshape the image
        x_random_reshaped = x[random_index].reshape((20, 20))

        # Display the image
        ax.imshow(x_random_reshaped, cmap='gray')

        # Predict using the Neural Network
        prediction = model.predict(x[random_index].reshape(1, 400))
        yhat = np.argmax(prediction)
        # If you need probability, you have to apply softmax
        prediction_probability = tf.nn.softmax(prediction)

        # Display the label above the image
        ax.set_title(f"{y[random_index, 0]},{yhat}")
        ax.set_axis_off()

    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()
