import numpy as np
import pandas as pd
import random as rd
from sklearn.model_selection import train_test_split

ERROR_LABEL_EMPTY="df_label_list is empty !"

def add_split_dataset(df_in: pd.DataFrame, split_timeframe: str = "Q", split_pattern: tuple = (60, 20, 20), clean_na: bool = True, fix_end_val: bool = True, random_split: bool = True) -> pd.DataFrame:
    """Add a column split_value in a time index dataframe,
    used to split a dataframe into train 0, validation 1, confirmation 2

    Args:
        df_in (pd.DataFrame): input dataframe, must be timestamp index
        split_timeframe (str, optional): timeframe used to split (H, D, M, Q, Y). Defaults to "Q".
        split_pattern (tuple, optional): split %, must be equal to 100. Defaults to (60, 20, 20).
        clean_na (bool, optional): if a dropna is done. Defaults to True.
        fix_end_val (bool, optional): if confirmation must be at the end of dataframe. Defaults to True.
        random_split (bool, optional) : If the split is random. If False, fix_end_val is useless. Defaults to True.

    Returns:
        pd.DataFrame: Same dataframe with a column splut_value at the end
    """

    df_split = df_in.copy()

    if clean_na:
        df_split.dropna(inplace=True)

    df_split['split_group_ut'] = pd.PeriodIndex(
        df_split.index, freq=split_timeframe)
    df_split['split_group_num'] = df_split.groupby(
        by=['split_group_ut']).ngroup()
    nb_groups = df_split['split_group_num'].max()+1
    nb_conf = (round(nb_groups*split_pattern[2]/100, 0)).astype(int)
    nb_val = (round(nb_groups*split_pattern[1]/100, 0)).astype(int)

    if (not (random_split)):
        df_split['split_value'] = np.where(
            df_split['split_group_num'] < nb_groups-nb_conf, 1, 2)
        df_split['split_value'] = np.where(
            df_split['split_group_num'] < nb_groups-nb_conf-nb_val, 0, df_split['split_value'])
    elif (fix_end_val):
        df_split['split_value'] = np.where(
            df_split['split_group_num'] < nb_groups-nb_conf, 0, 2)
        df_split['split_value'] = np.where(df_split['split_group_num'].isin(
            rd.sample(range(0, nb_groups-nb_conf-1), nb_val)), 1, df_split['split_value'])
    else:
        df_split['split_value'] = np.where(df_split['split_group_num'].isin(
            rd.sample(range(0, nb_groups-1), nb_conf)), 2, 0)
        df_split['split_value'] = np.where(df_split['split_group_num'].isin(rd.choices(
            (df_split['split_group_num'][df_split['split_value'] == 0]).unique(), k=nb_val)), 1, df_split['split_value'])

    df_split.drop(axis=1, columns=[
                  'split_group_ut', 'split_group_num'], inplace=True)
    return df_split


def split_df_by_split_value(df_in: pd.DataFrame, column_name: str = "split_value", drop_column: bool = True) -> list:
    """ split a df in 3 df (train, validation, confirmation) according a "split column"
    The values must be 0=train , 1=validation, 2=confirmation

    Args:
        df_in (pd.DataFrame): dataframe to split
        column_name (str, optional): name of the column containing the value used to split. Defaults to "split_value".
        drop_column (bool, optional): if it drops the cloumn column_name after split. Defaults to True.

    Raises:
        ValueError: if column_name is not found

    Returns:
        list: list of 3 dataframes [train, validation, confirmation]
    """
    df_tmp = df_in.copy()
    nb_split = (df_tmp[column_name].max())+1
    df_splitted = ["empty", ]*nb_split

    if column_name in df_in.columns:
        for i in range(0, nb_split):
            df_splitted[i] = df_in.loc[df_in[column_name] == i].copy(deep=True)
            if drop_column:
                df_splitted[i].drop(
                    axis=1, columns=[column_name], inplace=True)
    else:
        raise ValueError(f"column {column_name} not in dataframe !")

    return df_splitted


def split_df_by_label(df_in: pd.DataFrame, list_label: list, prefix_key: str = "df_", drop_na: bool = True) -> dict:
    """ split a dataset with n labels int a dictionary of n dataframes
    keys are prefixe_key+label

    Args:
        df_in (pd.DataFrame): the dataset to split
        df_label_list (list): list of label columns
        prefix_key (str, optional): prefix for the keys of dict. Defaults to "df_"
        drop_na (bool, optional): if a dropna is done on the new dataframe. Defaults to True.

    Raises:
        ValueError: if list of label is empty

    Returns:
        dict: a dictionnary containing one dataframe per label
    """
    dict_ret = dict()
    if len(list_label) > 0:
        for lab in list_label:
            df_tmp = df_in.copy()
            list_tmp = list_label.copy()
            list_tmp.remove(lab)
            df_tmp.drop(list_tmp, axis=1, inplace=True)
            if drop_na:
                df_tmp.dropna(inplace=True)
            dict_ret[prefix_key+lab] = df_tmp

    else:
        raise ValueError(ERROR_LABEL_EMPTY)

    return dict_ret


def split_df_by_label_strat(df_in: pd.DataFrame, list_label: list, prefix_key: str = "df_", drop_na: bool = True, split_timeframe: str = "Q", split_strat: tuple = (60, 20, 20), fix_end_val: bool = True, random_split: bool = True) -> dict:
    """ executes split_df_by_label, add_split_dataset and split_df_by_split_value on df_in
    returns a dict with all splitted dataframes (9 df)

    Args:
        df_in (pd.DataFrame): the dataset to split
        list_label (list): list of label columns
        prefix_key (str, optional): prefix for the keys of dict. Defaults to "df_". Defaults to "df_".
        drop_na (bool, optional): if a dropna is done on the new dataframe. Defaults to True.
        split_timeframe (str, optional): timeframe used to split (H, D, M, Q, Y). Defaults to "Q".
        split_strat (tuple, optional): split %, must be equal to 100. Defaults to (60, 20, 20).
        fix_end_val (bool, optional): if confirmation must be at the end of dataframe. Defaults to True.
        random_split (bool, optional) : If the split is random. If False, fix_end_val is useless. Defaults to True.

    Raises:
        ValueError: if list of label is empty

    Returns:
        dict: a dictionnary containing one dataframe per label per strat part, usually 9 dataframes 
    """
    dict_final = dict()
    if len(list_label) > 0:
        df_tmp = df_in.copy()
        dict_df_label = split_df_by_label(
            df_in=df_tmp, list_label=list_label, prefix_key=prefix_key, drop_na=drop_na)
        for key, value in dict_df_label.items():
            df_split = add_split_dataset(
                df_in=value, split_timeframe=split_timeframe, split_pattern=split_strat, fix_end_val=fix_end_val, random_split=random_split)
            df_train, df_val, df_conf = split_df_by_split_value(df_in=df_split)
            dict_final[key+'_train'] = df_train
            dict_final[key+'_valid'] = df_val
            dict_final[key+'_confirm'] = df_conf
    else:
        raise ValueError(ERROR_LABEL_EMPTY)

    return dict_final


def split_df_x_y(df_in: pd.DataFrame, str_label: str, list_features: list=None,  drop_na: bool = True) -> tuple:
    """ split a dataframe into a features datafame and the label serie

    Args:
        df_in (pd.DataFrame): dataframe to split
        str_label (str): name of the label
        list_features (list, optional): list of column names for features, if None drops str_label. Defaults to None.
        drop_na (bool, optional): if drop na before split. Defaults to True.

    Returns:
        tuple: (X dataframe, y serie)
    """

    df_tmp = df_in.copy()
    if drop_na:
        df_tmp.dropna(inplace=True)

    if list_features!=None:
        if len(list_features)>0:
            x_cols = df_tmp[list_features]
        else:
            raise ValueError(ERROR_LABEL_EMPTY)
    else:
        x_cols=df_tmp.drop(columns=[str_label])

    y_col = df_tmp[str_label]
    return x_cols, y_col

def remove_columns(df_in: pd.DataFrame, df_labels: pd.DataFrame, str_label: str="") -> pd.DataFrame:
    """Remove columns of a dataframe listed in an other dataframe except the one passed in input

    Args:
        df_in (pd.DataFrame): the input dataframe 
        df_labels (pd.DataFrame): a dataframe with 1 column listing columns name
        str_label (str): the name of a column in df_labels but we want to keep in the dataframe

    Returns:
        pd.DataFrame: the cleaned dataframe
    """

    lab_cols = df_labels[df_labels.columns[0]].to_list()
    lab_cols.remove(str_label)
    
    cols_to_keep=list(set(df_in.columns.tolist()).difference(lab_cols))

    return df_in[cols_to_keep]

def split_df_strat_lstm(df_in: pd.DataFrame, str_label: str, split_strat: tuple = (60, 20, 20), sequence_length: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare the dataset by performing stratified split into training, validation, and confirmation sets,
    and prepare the data for use with an LSTM model.
    Args:
        df_in  (pd.DataFrame): Input DataFrame with time index, features, and labels.
        str_label (str): Name of the label
        split_strat (tuple, optional): Split %, must be equal to 100. Defaults to (60, 20, 20).
        sequence_length (int): Length of feature sequences for LSTM.

    Returns:
        Tuple of NumPy arrays: X_train_lstm, y_train_lstm, X_valid_lstm, y_valid_lstm, X_conf_lstm, y_conf_lstm.
    """
    # Select feature columns and label column
    df_clean = df_in.dropna(inplace=False).sort_index()

    df_features,df_label=split_df_x_y(df_in=df_clean, str_label=str_label)

    # Perform stratified split to obtain train, valid, and test sets
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        df_features, df_label, train_size=split_strat[0]*0.01, stratify=df_label)
    x_valid, x_conf, y_valid, y_conf = train_test_split(
        x_remaining, y_remaining, train_size=split_strat[1] / (split_strat[1] + split_strat[2]), stratify=y_remaining)

    # Convert data to NumPy arrays
    x_train = x_train.values
    y_train = y_train.values
    x_valid = x_valid.values
    y_valid = y_valid.values
    x_conf = x_conf.values
    y_conf = y_conf.values

    # Prepare sequences for LSTM model
    x_train_lstm, y_train_lstm = prepare_sequences(
        x_train, y_train, sequence_length)
    x_valid_lstm, y_valid_lstm = prepare_sequences(
        x_valid, y_valid, sequence_length)
    x_conf_lstm, y_test_lstm = prepare_sequences(
        x_conf, y_conf, sequence_length)

    return x_train_lstm, y_train_lstm, x_valid_lstm, y_valid_lstm, x_conf_lstm, y_test_lstm


def prepare_sequences(np_x: np.ndarray, np_y: np.ndarray, sequence_length: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences of features and labels for LSTM model.
    Args:
        np_x (np.ndarray): Input features.
        np_y (np.ndarray): Input labels.
        sequence_length (int): Length of feature sequences for LSTM.

    Returns:
        Tuple of NumPy arrays: X_seq, y_seq.
    """

    num_samples = np_x.shape[0]

    x_seq = []
    y_seq = []

    # Generate sequences
    for i in range(num_samples - sequence_length + 1):
        x_seq.append(np_x[i:i + sequence_length])
        y_seq.append(np_y[i + sequence_length - 1])

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    # Convert labels to one-hot encoding
    y_seq = pd.get_dummies(y_seq).values

    return x_seq, y_seq

def prepare_sequences_with_df(df_in: pd.DataFrame,str_label:str, sequence_length: int = 10) -> tuple[np.ndarray, np.ndarray,pd.DataFrame]:
    """
    Prepare sequences of features and labels for LSTM model.
    Args:
        df_x (pd.DataFrame): Input features.
        str_label (str): Name of the label.
        sequence_length (int): Length of feature sequences for LSTM.

    Returns:
        Tuple of NumPy arrays and the dataframe associated: X_seq, y_seq, df.
    """
    df_tmp=df_in.copy()
    num_samples = df_tmp.shape[0]

    df_x,df_y=split_df_x_y(df_in=df_tmp,str_label=str_label,drop_na=False)

    x_seq = []
    y_seq = []
    df_ret=pd.DataFrame(columns=df_tmp.columns)

    # Generate sequences
    for i in range(num_samples - sequence_length + 1):
        idx=i + sequence_length - 1
        x_seq.append(df_x[i:idx+1])
        y_seq.append(df_y[idx])
        row=df_tmp.iloc[idx].to_frame().T
        df_ret=pd.concat([df_ret,row],ignore_index=False)

    x_seq = np.array(x_seq)
    y_seq = np.array(y_seq)

    # Convert labels to one-hot encoding
    y_seq = pd.get_dummies(y_seq).values

    return x_seq, y_seq, df_ret

def join_dataframes_backtest(df_candle: pd.DataFrame, df_score: pd.DataFrame, str_col_score: str, str_col_sl: str = None, str_col_tp: str = None, int_vol_def: int = 100000, fl_sl_def: float = None, fl_tp_def: float = None) -> pd.DataFrame:
    """Join a dataframe with candles data and a dataframe with score pridction to return a dataframe ready for the backtrader part

    Args:
        df_candle (pd.DataFrame): dataframe with candles data
        df_score (pd.DataFrame): dataframe with predictions
        str_col_score (str): column name of the score prediction
        str_col_sl (str, optional): column name of the SL. Defaults to None.
        str_col_tp (str, optional): column name of the TP. Defaults to None.
        int_vol_def (int, optional): default volume . Defaults to 100000.
        fl_sl_def (float, optional): default SL. Defaults to None.
        fl_tp_def (float, optional): default TP. Defaults to None.

    Returns:
        pd.DataFrame: a dataframe formatted for the backtrader backtest
    """
    df_candle_sel = df_candle.copy()
    df_candle_sel = df_candle_sel.loc[:, [
        'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME']]
    df_candle_sel.loc[:, 'VOLUME'] = np.where(df_candle_sel['VOLUME'].isnull() | (
        df_candle_sel['VOLUME'] == 0), int_vol_def, df_candle_sel['VOLUME'])

    df_score_sel = df_score.copy()
    df_score_sel = df_score_sel.loc[:, [str_col_score]]
    df_score_sel['SL'] = fl_sl_def
    if str_col_sl != None:
        df_score_sel['SL'] = df_score[str_col_sl]
    df_score_sel['TP'] = fl_tp_def
    if str_col_tp != None:
        df_score_sel['TP'] = df_score[str_col_tp]

    df_joined = df_candle_sel.join(df_score_sel, how='inner')
    df_joined.rename(columns={'OPEN': 'open', 'HIGH': 'high', 'LOW': 'low',
                     'CLOSE': 'close', 'VOLUME': 'volume'}, inplace=True)
    df_joined.rename(columns={
                     str_col_sl: 'SL', str_col_tp: 'TP', str_col_score: 'Predict'}, inplace=True)

    return df_joined


if __name__ == "__main__":
    nb_groups = 4
    nb_val = 20
    ut = "D"

    open, close, label1, label2, label3 = rd.choices(range(10, 20), k=nb_val), rd.choices(
        range(10, 20), k=nb_val), rd.choices(range(0, 2), k=nb_val), rd.choices(range(0, 2), k=nb_val), rd.choices(range(0, 2), k=nb_val)

    idx = pd.date_range("2022-01-01", periods=nb_val, freq=ut)
    frame = {'DATE': idx, 'OPEN': open, 'CLOSE': close,
             'LABEL_1': label1, 'LABEL_2': label2, 'LABEL_3': label3}

    df = pd.DataFrame(frame)
    df.set_index('DATE', inplace=True)

    # print(df)

    print("**********")
    lab = 'LABEL_1'
    df_lab = pd.DataFrame(['LABEL_1', 'LABEL_2', 'LABEL_3'])
    df_ok = remove_columns(df_in=df, df_labels=df_lab, str_label=lab)
    X_train, y_train, X_valid, y_valid, X_conf, y_conf = split_df_strat_lstm(
        df_in=df_ok, str_label=lab, split_strat=(70, 15, 15), sequence_length=5)

    print(f"SHAPES : {df.shape=} {df_ok.shape=} {X_train.shape=}  {y_train.shape=}  {X_valid.shape=} {y_valid.shape=} {X_conf.shape=} {y_conf.shape=} ")

    x_feat,y_lab,df_prep = prepare_sequences_with_df(df_in=df_ok,str_label=lab)

    print(f"DF SHAPES : {df_ok.shape=} {x_feat.shape=} {y_lab.shape=} {df_prep.shape=}" )
    print(df_prep)
