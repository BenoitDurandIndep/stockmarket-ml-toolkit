import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Iterable, Optional, Sequence, Tuple, Union


def lgbm_custom_penalty_score(
    y_true: Union[lgb.Dataset, np.ndarray],
    y_pred: np.ndarray,
) -> Tuple[str, float, bool]:
    """
    Custom penalty function for LightGBM to calculate the mean squared error between true and predicted labels.
    This function is used as a custom evaluation metric during the training of LightGBM models.
    Args:
        y_true (lgb.Dataset): True labels or dataset.
        y_pred (lgb.Dataset): Predicted labels or dataset.
    Returns:
        tuple: A tuple containing the name of the metric, the calculated error, and a boolean indicating if higher values are better.
    """
    # print(f"{y_true.shape=} {y_pred.shape=}")
    raw_true = y_true.get_label() if isinstance(y_true, lgb.Dataset) else y_true
    y_true_arr = np.asarray(raw_true)
    # Convert predicted probabilities to class labels
    if y_pred.ndim == 1:
        y_pred_labels = y_pred
    else:
        y_pred_labels = np.argmax(y_pred, axis=1)  # Convert predicted probabilities to class labels
    if y_true_arr.shape[0] != y_pred_labels.shape[0]:
        raise ValueError(f"Shape mismatch: y_true has shape {y_true_arr.shape} but y_pred_labels has shape {y_pred_labels.shape}")
  
    error = (y_true_arr - y_pred_labels) ** 2
    return 'custom_penalty', error.mean(), False  # Return the mean squared error (lower is better)

def print_eval_metric(env: lgb.callback.CallbackEnv) -> None:
    """
    Custom callback function to print evaluation metrics during LightGBM training.
    Args:
        env (lgb.callback.CallbackEnv): The callback environment containing information about the training process.
    """
    result = env.evaluation_result_list or []
    for item in result:
        if len(item) == 4:
            metric_name, dataset_name, metric_value, _ = item
        else:
            metric_name, dataset_name, metric_value, _, _ = item
        print(f"Iteration {env.iteration}: {dataset_name} - {metric_name} = {metric_value}")

def calculate_tree_depth(tree: dict) -> int:
    """
    Calculate the depth of a decision tree represented as a nested dictionary.
    Args:
        tree (dict): A dictionary representing the decision tree.
    Returns:
        int: The depth of the tree.
    """
    if 'leaf_index' in tree:
        return 0
    else:
        left_depth = calculate_tree_depth(tree['left_child'])
        right_depth = calculate_tree_depth(tree['right_child'])
        return 1 + max(left_depth, right_depth)

def expand_probabilities(df_in:pd.DataFrame, column_prefix:str, proba_column:str, num_classes:int=10) -> pd.DataFrame:
    """
    Expands a column of probabilities into separate columns for each class.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the probabilities column.
    column_prefix (str): The prefix for the new columns (e.g., 'predict_10_proba').
    proba_column (str): The name of the column containing the probabilities.
    num_classes (int): The number of classes (length of the probability array). Default is 10.
    If the probabilities are not in a list format, this function will raise an error.

    Returns:
    pd.DataFrame: The updated DataFrame with expanded probability columns.
    """
    for i in range(num_classes):
        df_in[f'{column_prefix}_{i}'] = df_in[proba_column].apply(lambda x, i=i: x[i]).round(5)
    return df_in

if __name__ == "__main__":
    print("Hello world")
    # Example usage
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.8, 0.2, 0.6, 0.4])
    metric_name, error, is_higher_better = lgbm_custom_penalty_score(y_true, y_pred)
    print(f"Metric: {metric_name}, Error: {error}, Higher is better: {is_higher_better}")
    # Example tree structure
    tree = {
        'split_feature': 0,
        'split_gain': 1.5,
        'threshold': 0.5,
        'left_child': {
            'leaf_index': 0,
            'leaf_value': 1.0
        },
        'right_child': {
            'split_feature': 1,
            'split_gain': 2.0,
            'threshold': 1.5,
            'left_child': {
                'leaf_index': 1,
                'leaf_value': 0.5
            },
            'right_child': {
                'leaf_index': 2,
                'leaf_value': 0.0
            }
        }
    }
    depth = calculate_tree_depth(tree)
    print(f"Tree depth: {depth}")

    # Dummy values for required arguments
    X_train = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_train = np.array([0, 1, 0, 1])
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {"objective": "binary", "verbosity": -1}
    model = lgb.train(params, train_data, num_boost_round=1)
    iteration = 10
    begin_iteration = 0
    end_iteration = 100
    evaluation_result_list = [('custom_penalty', 'train', 0.123, False)]

    # Create CallbackEnv instance
    env = lgb.callback.CallbackEnv(
        model=model,
        params=params,
        iteration=iteration,
        begin_iteration=begin_iteration,
        end_iteration=end_iteration,
        evaluation_result_list=evaluation_result_list
    )
    print_eval_metric(env)