import pandas as pd
import numpy as np


def report_best_scores(results, n_top=3):
    """_summary_

    Args:
        results (_type_): _description_
        n_top (int, optional): _description_. Defaults to 3.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(f"Mean validation score: {results['mean_test_score'][candidate]} (std: {results['std_test_score'][candidate]})")
            print(f"Parameters: {results['params'][candidate]}")
            print("")


if __name__ == "__main__":
    print("Hello world")