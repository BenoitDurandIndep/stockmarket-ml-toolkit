import numpy as np
import pandas as pd
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Bidirectional

RANDOM_KEY=28

def report_best_scores(results, n_top: int = 3):
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


def search_cv_fit_report(estimator, params, x_train, y_train,eval_set:dict={},  method: str = "grid", n_iter: int = 100, cv: int = 5, verbose: int = 1, n_jobs: int = 1, scoring: str = "neg_mean_squared_error", n_top: int = 3, refit: bool | str = True):
    """ Does a RandomizedSearchCV and reports best scores

    Args:
        estimator (_type_): estimator object, random forest, xgboost ...
        params (dict): Dictionnary with params
        x_train (pd.DataFrame) : training dataset
        y_train (serie) : label of training dataset
        eval_set (dict, optional) : evaluation set [(X,y)]. Defaults to None
        method (str, optional) : optimiszation method grid or random. Defaults to grid.
        n_iter (int, optional): nb of param settings that are sampled. Defaults to 100.
        cv (int, optional): nb folds. Defaults to 5.
        verbose (int, optional): verbosity (0-3). Defaults to 1.
        n_jobs (int, optional): n jobs in parallel. Defaults to 1.
        scoring (str, optional) : scoring method. Defaults to neg_mean_squared_error #squared_error, f1.
        n_top (int, optional): nb models showed. Defaults to 3.
        refit(bool|str, optional): Refit an estimator using the best found parameters on the whole dataset. Defaults to True

    Returns:
        estimator : fitted estimator with the best model
    """
    if method == "random":
        fitted = RandomizedSearchCV(estimator=estimator,
                                    param_distributions=params,
                                    n_iter=n_iter,
                                    cv=cv,
                                    verbose=verbose,
                                    random_state=RANDOM_KEY,
                                    n_jobs=n_jobs, scoring=scoring, error_score='raise', refit=refit)
    else:
        fitted = GridSearchCV(estimator=estimator,
                              param_grid=params,
                              cv=cv,
                              verbose=verbose,
                              n_jobs=n_jobs, scoring=scoring, error_score='raise', refit=refit)

    fitted.fit(x_train, y_train,eval_set=eval_set)

    if (verbose > 0):
        print(f"Accuracy train ({scoring}) :{fitted.score(x_train,y_train)}")

    if isinstance(refit,bool):
        report_best_scores(fitted.cv_results_, n_top)

    return fitted


def create_scikeras_lstm_model(layers: list, meta: dict,  dropout: float = 0.2, activation: str = 'tanh', optimizer: str = 'adam') -> Sequential:
    """
    Create an scikeras LSTM model with specified hyperparameters.

    Parameters:
        layers (list): List of layers to be added to the model. Each layer is a tuple with layer[0]=type (LSTM or Dense) and layer[1] number of neurons in the layer.
        window_size (int): Number of window_size in the input sequence.
        n_features_in_ (int): Number of input features.
        num_classes (int): Number of output classes.
        neurons (int): Number of neurons in the LSTM layer. Default is 64.
        dropout (float): Dropout rate between LSTM layers. Default is 0.2.
        activation (str): Activation function for the LSTM layer. Default is 'tanh'.
        optimizer (str): Optimizer used for training the model. Default is 'adam'.

    Returns:
        Sequential: The constructed LSTM model.

    Raises:
        ValueError: If the layer type is not supported. Must be LSTM or Dense.
    """

    n_features_in_ = meta["n_features_in_"]
    x_shape_ = meta["X_shape_"][2]
    n_classes_ = meta["n_classes_"]

    model = Sequential()

    for i, layer in enumerate(layers):
        if len(layer) == 1:
            layer_type = "LSTM"
            neurons = layer[0]
        else:
            layer_type = layer[0]
            neurons = layer[1]
        # print(f"{i=} {neurons=} {layer_type=} {layer=} ")
        if layer_type == "LSTM":
            if i == 0:
                model.add(LSTM(units=neurons, return_sequences=True, dropout=dropout,
                        activation=activation,  input_shape=(n_features_in_, x_shape_)))
            elif i == len(layers)-1:#last LSTM layer
                model.add(Bidirectional(
                    LSTM(units=neurons, return_sequences=False, activation=activation)))
            else:
                model.add(Bidirectional(LSTM(
                    units=neurons, return_sequences=True, dropout=dropout, activation=activation)))
        elif layer_type == "Dense":   
             model.add(Dense(units=neurons, activation=activation))
        else:
            raise ValueError(f"Layer type {layer[0]} not supported , must be LSTM or Dense.")

    model.add(Dense(units=n_classes_))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    
    return model


def create_sklearn_lstm_model(layers: list, input_dim: int, window_size: int, num_classes: int,  dropout: float = 0.2, activation: str = 'tanh', optimizer: str = 'adam') -> Sequential:
    """
    Create an SKLearn LSTM model with specified hyperparameters.

    Parameters:
        window_size (int): Number of window_size in the input sequence.
        input_dim (int): Number of input features.
        num_classes (int): Number of output classes.
        neurons (int): Number of neurons in the LSTM layer. Default is 64.
        dropout (float): Dropout rate between LSTM layers. Default is 0.2.
        activation (str): Activation function for the LSTM layer. Default is 'tanh'.
        optimizer (str): Optimizer used for training the model. Default is 'adam'.

    Returns:
        Sequential: The constructed LSTM model.
    """

    n_features_in_ = window_size
    x_shape_ = input_dim
    n_classes_ = num_classes

    model = Sequential()

    for i, neurons in enumerate(layers):
        print(f"{i=} {neurons=} {n_features_in_=} {x_shape_=} {n_classes_}")
        if i == 0:
            model.add(LSTM(units=neurons, return_sequences=True, dropout=dropout,
                      activation=activation,  input_shape=(n_features_in_, x_shape_)))
        elif i == len(layers)-1:
            model.add(Bidirectional(
                LSTM(units=neurons, return_sequences=False, activation=activation)))
        else:
            model.add(Bidirectional(LSTM(
                units=neurons, return_sequences=True, dropout=dropout, activation=activation)))
            
    model.add(Dense(units=n_classes_))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model



if __name__ == "__main__":
    
    meta = {"window_size": 10, "X_shape_": (None, 10, 5), "n_classes_": 3,"n_features_in_":5}
    model = create_scikeras_lstm_model(layers=[("LSTM",128), ("Dense",64), ("LSTM",64),("Dense",32), ("LSTM",32)], meta=meta)
    print(model.summary())


