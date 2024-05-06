import numpy as np
import pandas as pd
import random as rd
from typing import Dict
import pickle
from sklearn.preprocessing import PowerTransformer, StandardScaler, MinMaxScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE


def log_transform_label(df_in: pd.DataFrame, str_label: str, fl_fix: float = 1e-6) -> pd.DataFrame | float:
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

def add_class_by_lab_nb_lines(df_in: pd.DataFrame, str_label: str, nb_class: int = 10, bool_replace_label: bool = False, str_suffix_class: str = "_class") -> pd.DataFrame:
    """add a column str_label+str_suffix_class with an integer to split equally the label by number of lines
        using pd.qcut
    
    Args:
        df_in (pd.DataFrame): dataframe to use
        str_label (str): column with the label to split
        nb_class (int): number of classes needed
        bool_replace_label (bool, optional): True, replace the existing label by the new one. False add a new column with the new label. Defaults to True.
        str_suffix_class (str): suffix added to the label name for the new column. Defaults to "_class".

    Returns:
        pd.DataFrame: the dataframe with the str_label+str_suffix_class column
    """
    df_out = df_in.copy()

    new_lab = str_label + str_suffix_class

    df_out[new_lab] = pd.qcut(df_out[str_label], q=nb_class, labels=False)

    if bool_replace_label:
        df_out = df_out.drop(str_label, axis=1)
        df_out = df_out.rename(columns={new_lab: str_label})

    return df_out

def add_class_by_lab_value(df_in: pd.DataFrame, str_label: str, nb_class: int = 10, bool_replace_label: bool = False, str_suffix_class: str = "_class") -> pd.DataFrame:
    """add a column str_label+str_suffix_class with an integer to split equally the label by value
        using pd.cut
    
    Args:
        df_in (pd.DataFrame): dataframe to use
        str_label (str): column with the label to split
        nb_class (int): number of classes needed
        bool_replace_label (bool, optional): True, replace the existing label by the new one. False add a new column with the new label. Defaults to True.
        str_suffix_class (str): suffix added to the label name for the new column. Defaults to "_class".

    Returns:
        pd.DataFrame: the dataframe with the str_label+str_suffix_class column
    """
    df_out = df_in.copy()

    new_lab = str_label + str_suffix_class

    df_out[new_lab] = pd.cut(df_out[str_label], bins=nb_class, labels=False)

    if bool_replace_label:
        df_out = df_out.drop(str_label, axis=1)
        df_out = df_out.rename(columns={new_lab: str_label})

    return df_out


def add_lab_by_class(df_in: pd.DataFrame, str_label: str, categ: Dict[int, list], bool_replace_label: bool = False, str_suffix_class: str = "_class") -> pd.DataFrame:
    """add/replace a label for categorization based on the value of a continous label

    Args:
        df_in (pd.DataFrame): dataframe to use
        str_label (str): column with the existing label
        categ (Dict[int, list]): limits for the categories eg {0:[5,10],1:[0,5]}
        bool_replace_label (bool, optional): True, replace the existing label by the new one. False add a new column with the new label. Defaults to True.
        str_suffix_class (str, optional): Suffix for the new label. Defaults to "_class".

    Returns:
        pd.DataFrame: the new dataframe
    """
    df_out = df_in.copy()

    new_lab = str_label + str_suffix_class

    for key, value in categ.items():
        class_min = value[0]
        class_max = value[1]
        df_out.loc[(df_out[str_label] >= class_min) & (
            df_out[str_label] <= class_max), new_lab] = key

    if bool_replace_label:
        df_out = df_out.drop(str_label, axis=1)
        df_out = df_out.rename(columns={new_lab: str_label})

    return df_out


def class_oversampler(df_in: pd.DataFrame, str_label: str, str_method: str = "RandomOverSampler", str_strat: str = "auto", dict_strat: dict = None) -> pd.DataFrame:
    """Oversample a dataframe

    Args:
        df_in (pd.DataFrame): dataframe to undersample
        str_label (str): column with the classification label
        str_method (str, optional): the method used to oversample (SMOTE). Defaults to "RandomUnderSampler".
        strat (str,optional) : sampling strategy for the method. Defaults to "auto"
        dict_strat(dict,optional) : if filled, When dict, the keys correspond to the targeted classes. The values correspond to the desired number of samples for each targeted class. to Defaults to None

    Returns:
        pd.DataFrame: the dataframe oversampled
    """
    df_out = df_in.copy()

    X = df_out.drop(str_label, axis=1, inplace=False)
    y = df_out[str_label]

    strat = dict_strat if dict_strat is not None else str_strat

    if str_method == "SMOTE":
        method = SMOTE(sampling_strategy=strat)
    else:
        method = RandomOverSampler(sampling_strategy=strat)

    x_samp, y_samp = method.fit_resample(X, y)

    df_resampled = pd.DataFrame(y_samp, columns=[str_label])
    df_resampled = pd.concat([x_samp, df_resampled], axis=1)

    return df_resampled


def class_undersampler(df_in: pd.DataFrame, str_label: str, str_method: str = "RandomUnderSampler", str_strat: str = "auto", dict_strat: dict = None) -> pd.DataFrame:
    """Undersample a dataframe

    Args:
        df_in (pd.DataFrame): dataframe to undersample
        str_label (str): column with the classification label
        str_method (str, optional): the method used to undersample (NearMiss,TomekLinks). Defaults to "RandomUnderSampler".
        strat (str,optional) : sampling strategy for the method. Defaults to "auto"
        dict_strat(dict,optional) : if filled, When dict, the keys correspond to the targeted classes. The values correspond to the desired number of samples for each targeted class. to Defaults to None

    Returns:
        pd.DataFrame: the dataframe undersampled
    """
    df_out = df_in.copy()

    X = df_out.drop(str_label, axis=1, inplace=False)
    y = df_out[str_label]

    strat = dict_strat if dict_strat is not None else str_strat

    if str_method == "NearMiss":
        method = NearMiss(sampling_strategy=strat)
    elif str_method == "TomekLinks":
        method = TomekLinks(sampling_strategy=strat)
    else:
        method = RandomUnderSampler(sampling_strategy=strat)

    x_samp, y_samp = method.fit_resample(X, y)
    # get index from previous df
    x_samp.index = X.index[method.sample_indices_]

    df_resampled = x_samp.join(df_out.loc[:, [str_label]], how='inner')

    return df_resampled


def reg_undersampler_by_class(df_in: pd.DataFrame, str_label: str, str_method: str = "RandomUnderSampler", nb_class: int = 10, str_strat: str = "auto") -> pd.DataFrame:
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

    df_class = add_class_by_lab_value(df_in=df_out, str_label=str_label,
                         nb_class=nb_class, str_suffix_class=suffix_class)

    lab_class = str_label+suffix_class
    df_class.drop(str_label, axis=1, inplace=True)
    X = df_class.drop(lab_class, axis=1)
    y = df_class[lab_class]

    if str_method == "NearMiss":
        method = NearMiss(sampling_strategy=str_strat)
    elif str_method == "TomekLinks":
        method = TomekLinks(sampling_strategy=str_strat)
    else:
        method = RandomUnderSampler(sampling_strategy=str_strat)
    x_samp, y_samp = method.fit_resample(X, y)
    # get index from previous df
    x_samp.index = X.index[method.sample_indices_]

    df_resampled = x_samp.join(df_out.loc[:, [str_label]], how='inner')

    return df_resampled


def yeo_johnson_transform_col(df_in: pd.DataFrame, str_col: str) -> pd.DataFrame | PowerTransformer:
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


def normalize_df(df_in: pd.DataFrame, str_label: str=None,tuple_ft_range:tuple=(0,1)) -> pd.DataFrame | MinMaxScaler:
    """Normalize the features of a dataset with StandaedScaler

    Args:
        df_in (pd.DataFrame): The dataset to normalize
        str_label (str): The name of the label Default None
        tuple_ft_range (tuple) : the range of the normalize Default (0,1)

    Returns:
        pd.DataFrame: The normalized dataset
    """
    df_out = df_in.copy()
    if str_label is None:
        col_to_norm = df_out.columns
    else:
        col_to_norm = df_out.columns.drop(str_label)

    scaler = MinMaxScaler(feature_range=tuple_ft_range)

    df_out[col_to_norm] = scaler.fit_transform(df_out[col_to_norm])

    return df_out, scaler

def normalize_df_scaler(df_in: pd.DataFrame, str_label: str,scaler:MinMaxScaler) -> pd.DataFrame:
    """Normalize the features of a dataset with MinMaxScaler

    Args:
        df_in (pd.DataFrame): The dataset to normalize
        str_label (str): The name of the label
        scaler (MinMaxScaler): The MinMaxScaler used for the normalization

    Returns:
        pd.DataFrame: The normalized dataset
    """
    df_out = df_in.copy()
    col_to_norm = df_out.columns.drop(str_label)

    df_out[col_to_norm] = scaler.transform(df_out[col_to_norm])

    return df_out

def normalize_df_inverse(df_in: pd.DataFrame, str_label: str,scaler:MinMaxScaler) -> pd.DataFrame:
    """Inverse the normalization of a dataset with MinMaxScaler

    Args:
        df_in (pd.DataFrame): The dataset to normalize
        str_label (str): The name of the label
        scaler (MinMaxScaler): The MinMaxScaler used for the normalization

    Returns:
        pd.DataFrame: The normalized dataset
    """
    df_out = df_in.copy()
    col_to_norm = df_out.columns.drop(str_label)

    df_out[col_to_norm] = scaler.inverse_transform(df_out[col_to_norm])

    return df_out

def standardize_df(df_in: pd.DataFrame, str_label: str) -> pd.DataFrame | StandardScaler:
    """Standardize the features of a dataset with StandaedScaler

    Args:
        df_in (pd.DataFrame): The dataset to normalize
        str_label (str): The name of the label

    Returns:
        pd.DataFrame: The normalized dataset
    """
    df_out = df_in.copy()
    col_to_norm = df_out.columns.drop(str_label)

    scaler = StandardScaler()

    df_out[col_to_norm] = scaler.fit_transform(df_out[col_to_norm])

    return df_out, scaler

def save_transformer(transformer: PowerTransformer, filename: str):
    """Save a PowerTransformer

    Args:
        transformer (PowerTransformer): transformer to save
        filename (str): path and name of the file
    """
    with open(filename, 'wb') as f:
        pickle.dump(transformer, f)


def load_transformer(filename: str) -> PowerTransformer:
    """Load a PowerTransformer

    Args:
        filename (str): path and name of the file

    Returns:
        PowerTransformer: The PowerTransformer
    """
    with open(filename, 'rb') as f:
        transformer = pickle.load(f)
    return transformer


def save_scaler(scaler: StandardScaler, filename: str):
    """Save a StandardScaler

    Args:
        scalerer (StandardScaler): scaler to save
        filename (str): path and name of the file
    """
    with open(filename, 'wb') as f:
        pickle.dump(scaler, f)


def load_scaler(filename: str) -> StandardScaler:
    """Load a StandardScaler

    Args:
        filename (str): path and name of the file

    Returns:
        StandardScaler: The StandardScaler
    """
    with open(filename, 'rb') as f:
        scaler = pickle.load(f)
    return scaler


if __name__ == "__main__":
    nb_groups = 20
    nb_val = 1000
    ut = "D"

    open, close, label1, label2, label3 = rd.choices(range(10, 20), k=nb_val), rd.choices(
        range(10, 20), k=nb_val), rd.choices(range(-100, 100), k=nb_val), rd.choices(range(0, 2), k=nb_val), rd.choices(range(0, 2), k=nb_val)

    idx = pd.date_range("2022-01-01", periods=nb_val, freq=ut)
    frame = {'DATE': idx, 'OPEN': open, 'CLOSE': close,
             'LABEL_1': label1, 'LABEL_2': label2, 'LABEL_3': label3}

    df = pd.DataFrame(frame)
    df.set_index('DATE', inplace=True)
    print(df.describe())

    df_out=add_class_by_lab_nb_lines(df_in=df,str_label='LABEL_1',nb_class=5)
    print(df_out['LABEL_1_class'].value_counts().sort_index())
