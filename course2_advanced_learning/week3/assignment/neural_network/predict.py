import numpy as np
import tensorflow as tf

from course2_advanced_learning.week3.assignment.neural_network.utils import eval_category_error_nn


def predict_on_nn_model(model, model_name, x_train, y_train, x_cv, y_cv):
    model_predict = lambda xl: np.argmax(tf.nn.softmax(model.predict(xl)).numpy(), axis=1)

    training_cerr_complex = eval_category_error_nn(y_train, model_predict(x_train))
    cv_cerr_complex = eval_category_error_nn(y_cv, model_predict(x_cv))
    print(f"categorization error, training, {model_name}: {training_cerr_complex:0.3f}")
    print(f"categorization error, cv,       {model_name}: {cv_cerr_complex:0.3f}")

    return training_cerr_complex, cv_cerr_complex
