import numpy as np
import pandas as pd
import random as rd
import pickle
from sklearn.preprocessing import PowerTransformer
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from typing import Union


def log_transform_label(df_in: pd.DataFrame, str_label: str, fl_fix: float = 1e-6) -> Union[pd.DataFrame, float]:
    """logarithm transform the column str_label

    Args:
        df_in (pd.DataFrame): dataframe to use
        str_label (str): column with the value to transform
        fl_fix (float) : fixed value added to the shift to avoid infinity val. Defaults to 1e-6

    Returns:
        pd.DataFrame: the transformed dataframe
        float : the shift value
    """
    df_out = df_in.copy()
    shift_value = abs(df_out[str_label].min()) + fl_fix
    df_out[str_label] = np.log(df_out[str_label]+shift_value)
    return df_out, shift_value


def exp_transform_label(df_in: pd.DataFrame, str_label: str, fl_shift: float = 0) -> pd.DataFrame:
    """exponentiel transform the column str_label

    Args:
        df_in (pd.DataFrame): dataframe to use
        str_label (str): column with the value to transform
        fl_shift (float) : shift value given by log_transform_label. Defaults 0

    Returns:
        pd.DataFrame: the transformed dataframe
    """
    df_out = df_in.copy()
    df_out[str_label] = np.exp(df_out[str_label]+fl_shift)
    return df_out


def range_undersampler(df_in: pd.DataFrame, str_label: str, min_val: float, max_val: float, clean_rate: float) -> pd.DataFrame:
    """ randomly drop some lines of a dataframe if the value of the specified column is between min_val and max_val

    Args:
        df_in (pd.DataFrame): dataframe to under sample
        str_label (str): column with the value to check
        min_val (float): min value of the range
        max_val (float): max value of the range
        clean_rate (float): % of drop 1.0 for 100%

    Returns:
        pd.DataFrame: the undersampled dataframe 
    """
    mask = (df_in[str_label] >= min_val) & (df_in[str_label] <= max_val)

    # Proba for cleaning a line
    prob_clean = np.random.random(size=len(df_in))
    prob_masked = prob_clean <= clean_rate
    rows_to_remove = prob_masked & mask

    df_cleaned = df_in[~rows_to_remove]

    return df_cleaned


def clipping_col(df_in: pd.DataFrame, str_col: str, min_val: float, max_val: float) -> pd.DataFrame:
    """ Clip extreme values of a column

    Args:
        df_in (pd.DataFrame): dataframe to clip
        str_col (str): column with the value to check
        min_val (float): min value of the range
        max_val (float): max value of the range

    Returns:
        pd.DataFrame: the clipped dataframe
    """
    df_out = df_in.copy()

    df_out.loc[df_out[str_col] < min_val, str_col] = min_val
    df_out.loc[df_out[str_col] > max_val, str_col] = max_val

    return df_out


def add_class(df_in: pd.DataFrame, str_label: str, nb_class: int = 10, str_suffix_class: str = "_class") -> pd.DataFrame:
    """add a column str_label+str_suffix_class with an integer to split equally the label

    Args:
        df_in (pd.DataFrame): dataframe to use
        str_label (str): column with the label to split
        nb_class (int): number of classes needed
        str_suffix_class (str): suffix added to the label name for the new column

    Returns:
        pd.DataFrame: the dataframe with the str_label+str_suffix_class column
    """
    df_out = df_in.copy()

    label_range = df_out[str_label].max() - df_out[str_label].min()
    class_size = label_range / nb_class

    df_out[str_label + str_suffix_class] = ((df_out[str_label] -
                                             df_out[str_label].min()) // class_size).astype(int)

    return df_out


def reg_undersampler_by_class(df_in: pd.DataFrame, str_label: str, str_method: str = "RandomUnderSampler", nb_class: int = 10, strat: str = "auto") -> pd.DataFrame:
    """Add a classification label to a dataframe and then undersample on the dataframe

    Args:
        df_in (pd.DataFrame): dataframe to undersample
        str_label (str): column with the regression label
        str_method (str, optional): the method used to undersample (NearMiss,TomekLinks). Defaults to "RandomUnderSampler".
        nb_class (int, optional): number of classes needed. Defaults to 10.
        strat (str,optional) : sampling strategy for the method. Defaults to "auto"

    Returns:
        pd.DataFrame: the dataframe undersampled
    """
    suffix_class = "_class"
    df_out = df_in.copy()

    df_class = add_class(df_in=df_out, str_label=str_label,
                         nb_class=nb_class, str_suffix_class=suffix_class)

    lab_class = str_label+suffix_class
    df_class.drop(str_label, axis=1, inplace=True)
    X = df_class.drop(lab_class, axis=1)
    y = df_class[lab_class]

    if str_method == "NearMiss":
        method = NearMiss(sampling_strategy=strat)
    elif str_method == "TomekLinks":
        method = TomekLinks(sampling_strategy=strat)
    else:
        method = RandomUnderSampler(sampling_strategy=strat)
    X_samp, y_samp = method.fit_resample(X, y)
    # get index from previous df
    X_samp.index = X.index[method.sample_indices_]

    df_resampled = X_samp.join(df_out.loc[:, [str_label]], how='inner')

    return df_resampled


def yeo_johnson_transform_col(df_in: pd.DataFrame, str_col: str) -> Union[pd.DataFrame, PowerTransformer]:
    """Transform a column of a dataframe with Yeo-Johnson

    Args:
        df_in (pd.DataFrame): dataframe to use
        str_col (str): name of the column to transform

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df_out = df_in.copy()
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df_out[str_col] = pt.fit_transform(
        df_out[str_col].values.reshape(-1, 1)).flatten()
    return df_out, pt


def yeo_johnson_transform_inverse_col(df_in: pd.DataFrame, str_col: str, pt: PowerTransformer) -> pd.DataFrame:
    """Inverse the transformation a column of a dataframe with Yeo-Johnson

    Args:
        df_in (pd.DataFrame): dataframe to use
        str_col (str): name of the column to transform
        pt (PowerTransformer): Powertransformer used by the transform function

    Returns:
        pd.DataFrame: transformed dataframe
    """
    df_out = df_in.copy()
    df_out[str_col] = pt.inverse_transform(
        df_out[str_col].values.reshape(-1, 1)).flatten()
    return df_out

def save_transformer(transformer:PowerTransformer, filename:str):
    """Save a PowerTransformer

    Args:
        transformer (PowerTransformer): transformer to save
        filename (str): path and name of the file
    """
    with open(filename, 'wb') as f:
        pickle.dump(transformer, f)

def load_transformer(filename:str)->PowerTransformer:
    """Load a PowerTransformer

    Args:
        filename (str): path and name of the file

    Returns:
        PowerTransformer: The PowerTransformer
    """
    with open(filename, 'rb') as f:
        transformer = pickle.load(f)
    return transformer    

if __name__ == "__main__":
    nb_groups = 20
    nb_val = 1000
    ut = "D"

    open, close, label1, label2, label3 = rd.choices(range(10, 20), k=nb_val), rd.choices(
        range(10, 20), k=nb_val), rd.choices(range(0, 2), k=nb_val), rd.choices(range(0, 2), k=nb_val), rd.choices(range(0, 2), k=nb_val)

    idx = pd.date_range("2022-01-01", periods=nb_val, freq=ut)
    frame = {'DATE': idx, 'OPEN': open, 'CLOSE': close,
             'LABEL_1': label1, 'LABEL_2': label2, 'LABEL_3': label3}

    df = pd.DataFrame(frame)
    df.set_index('DATE', inplace=True)
    print(df.describe())
    df_sampled = reg_undersampler_by_class(
        df_in=df, str_label='LABEL_1', nb_class=20)
    print(df_sampled.describe())
