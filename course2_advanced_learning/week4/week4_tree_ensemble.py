import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 55  ## We will pass it to every sklearn call so we ensure reproducibility


def load_dataset():
    dataset = pd.read_csv("data/heart.csv")

    # One-hot encoding
    category_features = [
        'Sex',
        'ChestPainType',
        'RestingECG',
        'ExerciseAngina',
        'ST_Slope'
    ]
    # This will change categories like Sex from String[M/F] to 2 booleans (Sex_M and Sex_F)
    dataset = pd.get_dummies(
        data=dataset,
        prefix=category_features,
        columns=category_features
    )

    target_column_name = 'HeartDisease'
    feature_names = [x for x in dataset.columns if x not in target_column_name]

    x, y = dataset[feature_names], dataset[target_column_name]

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=RANDOM_STATE)

    return x_train, x_val, y_train, y_val


def choose_optimal_samples_split_for_decision_tree(x_train, y_train, x_val, y_val):
    min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
    accuracy_list_train = []
    accuracy_list_val = []
    for min_samples_split in min_samples_split_list:
        model = DecisionTreeClassifier(
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)

        predictions_train = model.predict(x_train)
        predictions_val = model.predict(x_val)

        accuracy_train = accuracy_score(predictions_train, y_train)
        accuracy_val = accuracy_score(predictions_val, y_val)

        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('min_samples_split')
    plt.ylabel('accuracy')
    plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train', 'Validation'])
    plt.show()


def choose_optimal_max_depth_for_decision_tree(x_train, y_train, x_val, y_val):
    max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]

    accuracy_list_train = []
    accuracy_list_val = []

    for max_depth in max_depth_list:
        model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)

        predictions_train = model.predict(x_train)
        predictions_val = model.predict(x_val)

        accuracy_train = accuracy_score(predictions_train, y_train)
        accuracy_val = accuracy_score(predictions_val, y_val)

        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train', 'Validation'])
    plt.show()


def choose_optimal_samples_split_for_random_forest(x_train, y_train, x_val, y_val):
    min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]

    accuracy_list_train = []
    accuracy_list_val = []
    for min_samples_split in min_samples_split_list:
        model = RandomForestClassifier(
            min_samples_split=min_samples_split,
            random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)

        predictions_train = model.predict(x_train)
        predictions_val = model.predict(x_val)

        accuracy_train = accuracy_score(predictions_train, y_train)
        accuracy_val = accuracy_score(predictions_val, y_val)

        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('min_samples_split')
    plt.ylabel('accuracy')
    plt.xticks(ticks=range(len(min_samples_split_list)), labels=min_samples_split_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train', 'Validation'])
    plt.show()


def choose_optimal_max_depth_for_random_forest(x_train, y_train, x_val, y_val):
    max_depth_list = [2, 4, 8, 16, 32, 64, None]

    accuracy_list_train = []
    accuracy_list_val = []
    for max_depth in max_depth_list:
        model = RandomForestClassifier(
            max_depth=max_depth,
            random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)

        predictions_train = model.predict(x_train)
        predictions_val = model.predict(x_val)

        accuracy_train = accuracy_score(predictions_train, y_train)
        accuracy_val = accuracy_score(predictions_val, y_val)

        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('max_depth')
    plt.ylabel('accuracy')
    plt.xticks(ticks=range(len(max_depth_list)), labels=max_depth_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train', 'Validation'])
    plt.show()


def choose_optimal_estimators_count_for_random_forest(x_train, y_train, x_val, y_val):
    n_estimators_list = [10, 50, 100, 500]

    accuracy_list_train = []
    accuracy_list_val = []
    for n_estimators in n_estimators_list:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=RANDOM_STATE
        )
        model.fit(x_train, y_train)

        predictions_train = model.predict(x_train)
        predictions_val = model.predict(x_val)

        accuracy_train = accuracy_score(predictions_train, y_train)
        accuracy_val = accuracy_score(predictions_val, y_val)

        accuracy_list_train.append(accuracy_train)
        accuracy_list_val.append(accuracy_val)

    plt.title('Train x Validation metrics')
    plt.xlabel('n_estimators')
    plt.ylabel('accuracy')
    plt.xticks(ticks=range(len(n_estimators_list)), labels=n_estimators_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(['Train', 'Validation'])
    plt.show()


def train_xgboost_model(x_train, y_train, x_val, y_val):
    # It will look for best set of trees - up to 500. In this case it should stop around 16 tree
    xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.1, verbosity=1, random_state=RANDOM_STATE,
                              early_stopping_rounds=10)
    xgb_model.fit(x_train, y_train, eval_set=[(x_val, y_val)])
    print(f"XGBoost stopped at {xgb_model.best_iteration} iteration")

    return xgb_model


if __name__ == '__main__':
    x_train, x_val, y_train, y_val = load_dataset()

    choose_optimal_samples_split_for_decision_tree(x_train, y_train, x_val, y_val)
    choose_optimal_max_depth_for_decision_tree(x_train, y_train, x_val, y_val)
    # Params chosen to have max validation accuracy. When multiple values have the same,
    # choose the one with lower training accuracy to prevent overfitting
    decision_tree_model = DecisionTreeClassifier(
        min_samples_split=50,
        max_depth=3,
        random_state=RANDOM_STATE
    ).fit(x_train, y_train)
    print(
        f"Decision tree metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(x_train), y_train):.4f}")
    print(
        f"Decision tree metrics validation:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(x_val), y_val):.4f}")

    choose_optimal_samples_split_for_random_forest(x_train, y_train, x_val, y_val)
    choose_optimal_max_depth_for_random_forest(x_train, y_train, x_val, y_val)
    choose_optimal_estimators_count_for_random_forest(x_train, y_train, x_val, y_val)
    # Params chosen to have max validation accuracy. When multiple values have the same,
    # choose the one with lower training accuracy to prevent overfitting
    random_forest_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        min_samples_split=10
    ).fit(x_train, y_train)
    print(
        f"Random forest metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(x_train), y_train):.4f}")
    print(
        f"Random forest metrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(x_val), y_val):.4f}")

    # XGBoost needs training and Cross validation data. Let's use 80% to train and 20% to eval
    n = int(len(x_train) * 0.8)
    x_train_fit, x_train_eval, y_train_fit, y_train_eval = x_train[:n], x_train[n:], y_train[:n], y_train[n:]
    xgboost_model = train_xgboost_model(x_train_fit, y_train_fit, x_train_eval, y_train_eval)
    # It actually finds a very similar accuracy to Random Forest without you having to manually look at hyperparams
    print(
        f"XGBoost metrics train:\n\tAccuracy score: {accuracy_score(xgboost_model.predict(x_train), y_train):.4f}")
    print(f"XGBoost metrics test:\n\tAccuracy score: {accuracy_score(xgboost_model.predict(x_val), y_val):.4f}")
