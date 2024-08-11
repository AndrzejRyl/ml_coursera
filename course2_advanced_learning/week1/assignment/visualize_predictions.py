import numpy as np
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
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0

        # Display the label above the image
        ax.set_title(f"{y[random_index, 0]},{yhat}")
        ax.set_axis_off()
    fig.suptitle("Label, yhat", fontsize=16)
    plt.show()

    all_predictions = []
    for i in range(len(y)):
        prediction = model.predict(x[i].reshape(1, 400))
        if prediction >= 0.5:
            yhat = 1
        else:
            yhat = 0
        all_predictions.append(yhat)

    all_predictions = np.array(all_predictions).T
    all_predictions = np.expand_dims(all_predictions, axis=-1)

    errors = np.where(y != all_predictions)
    number_of_errors = len(errors[0])
    print("Number of errors: %s" % number_of_errors)

    if number_of_errors > 0:
        fig = plt.figure(figsize=(1, 1))
        fig.tight_layout(pad=0.1, rect=[0, 0.03, 1, 0.92])

        first_error_index = errors[0][0]
        x_random_reshaped = x[first_error_index].reshape((20, 20))
        plt.imshow(x_random_reshaped, cmap='gray')
        plt.title(f"{y[first_error_index, 0]}, {all_predictions[first_error_index, 0]}")
        plt.axis('off')

        fig.suptitle("Error example", fontsize=16)
        plt.show()
