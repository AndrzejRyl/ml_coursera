import numpy as np


def load_data():
    x_train = np.array([[1, 1, 1],
                        [0, 0, 1],
                        [0, 1, 0],
                        [1, 0, 1],
                        [1, 1, 1],
                        [1, 1, 0],
                        [0, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0],
                        [0, 1, 0]])

    y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

    return x_train, y_train


def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def split_indices(x, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that
    have that feature = 1 and the right node those that have the feature = 0
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i, x in enumerate(x):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def weighted_entropy(x, y, left_indices, right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices) / len(x)
    w_right = len(right_indices) / len(x)
    p_left = sum(y[left_indices]) / len(left_indices)
    p_right = sum(y[right_indices]) / len(right_indices)

    result = w_left * entropy(p_left) + w_right * entropy(p_right)
    return result


def information_gain(x, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(x, y, left_indices, right_indices)
    return h_node - w_entropy


def build_tree_recursive(x, y, name, max_depth, current_depth, global_indices):
    if max_depth == current_depth:
        print(f"\t\t{name} leaf nodes: {global_indices}")
        return

    information_gains = []
    for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
        left_indices, right_indices = split_indices(x, i)
        if len(left_indices) > 0 and len(right_indices) > 0:
            i_gain = information_gain(x, y, left_indices, right_indices)
        else:
            i_gain = 0
        information_gains.append(i_gain)

    best_information_gain_idx = information_gains.index(max(information_gains))
    left_indices, right_indices = split_indices(x, best_information_gain_idx)

    print(f"Depth {current_depth}, {name}. Split on feature: {best_information_gain_idx}")

    build_tree_recursive(
        x=np.array([x_elem for idx, x_elem in enumerate(x) if idx in left_indices]),
        y=np.array([y_elem for idx, y_elem in enumerate(y) if idx in left_indices]),
        name="Left",
        max_depth=max_depth,
        current_depth=current_depth + 1,
        global_indices=[global_indices[index] for index in left_indices],
    )

    build_tree_recursive(
        x=np.array([x_elem for idx, x_elem in enumerate(x) if idx in right_indices]),
        y=np.array([y_elem for idx, y_elem in enumerate(y) if idx in right_indices]),
        name="Right",
        max_depth=max_depth,
        current_depth=current_depth + 1,
        global_indices=[global_indices[index] for index in right_indices],
    )


if __name__ == '__main__':
    x_train, y_train = load_data()

    build_tree_recursive(
        x=x_train,
        y=y_train,
        name="Root",
        max_depth=2,
        current_depth=0,
        global_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
