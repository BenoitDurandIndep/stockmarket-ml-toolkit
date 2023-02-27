import numpy as np
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb


def report_best_scores(results, n_top:int =3):
    """_summary_

    Args:
        results (_type_): croww validation result (cv_results_)
        n_top (int, optional): nb models showed. Defaults to 3.
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(f"Model with rank: {i}")
            print(
                f"Mean validation score: {results['mean_test_score'][candidate]} (std: {results['std_test_score'][candidate]})")
            print(f"Parameters: {results['params'][candidate]}")
            print("")


def randomizedsearch_cv_fit_report(estimator, param_distributions, x_train, y_train, random_state:int, n_iter:int=100, cv:int=5, verbose:int=1, n_jobs:int=1,scoring:str="neg_mean_squared_error", n_top:int=3):
    """ Does a RandomizedSearchCV and reports best scores

    Args:
        estimator (_type_): estimator object, random forest, xgboost ...
        param_distributions (dict): Dictionnary with params
        x_train (pd.DataFrame) : training dataset
        y_train (serie) : label of training dataset
        random_state (int): random state
        n_iter (int, optional): nb of param settings that are sampled. Defaults to 100.
        cv (int, optional): nb folds. Defaults to 5.
        verbose (int, optional): verbosity (0-3). Defaults to 1.
        n_jobs (int, optional): n jobs in parallel. Defaults to 1.
        scoring (str, optional) : scoring method. Defaults to neg_mean_squared_error.
        n_top (int, optional): nb models showed. Defaults to 3.

    Returns:
        estimator : fitted estimator with the best model
    """

    fitted = RandomizedSearchCV(estimator=estimator,
                                param_distributions=param_distributions,
                                n_iter=n_iter,
                                cv=cv,
                                verbose=verbose,
                                random_state=random_state,
                                n_jobs=n_jobs,scoring='neg_mean_squared_error')

    fitted.fit(x_train,y_train)

    if(verbose>0):
        print(f"Accuracy train (neg_mean_squared_error) :{fitted.score(x_train,y_train)}")

    report_best_scores(fitted.cv_results_, n_top)

    return fitted


if __name__ == "__main__":
    print("No test for this module yet.")
