import pandas as pd
import ta
from sqlalchemy import engine
from sqlite_io import get_connection, get_candles_to_df, get_ind_for_dts, get_ind_list_by_type_for_dts, get_ind_list_for_model,get_header_for_model

KEY_WORDS_LIST = ["CLOSE", "OPEN", "HIGH", "LOW", "IND", "VOLUME"]
DEFAULT_INDIC_SEP = "$$"


def get_indicator_value(df_in: pd.DataFrame, indic_code: str, sep: str = DEFAULT_INDIC_SEP) -> pd.Series:
    """ calculates and returns an indicator serie 
    Column used in the code must be surounded by sep

    Example of indic_code : ta.trend.SMAIndicator(close=$$CLOSE$$,window=).sma_indicator()
    or ($$CLOSE$$-$$IND$$!sma20$$IND$$)/$$IND$$!sma20$$IND$$

    Args:
        df_in (pd.DataFrame): Dataframe with the data needed for calculation
        indic_code (str): the code of the indicator 
        sep (str, optional): sep to split key words in the code. Defaults to "$$".

    Returns:
        pd.Series: the calculated indicator
    """
    filtered_df = pd.DataFrame()
    exec_code = indic_code
    tab_ind = exec_code.split(sep)
    filtered_cols = [col for col in tab_ind if col in KEY_WORDS_LIST]
    for col in filtered_cols:
        if col == "IND":  # case indicator based on an other indicator
            for code in tab_ind:
                if code.startswith("!"):
                    code = code[1:]
                    filtered_df[code] = df_in[code]
                    exec_code = exec_code.replace(
                        f"{sep}{col}{sep}!{code}{sep}{col}{sep}", f"filtered_df['{code}']")

        else:
            filtered_df[col] = df_in[col]
            exec_code = exec_code.replace(
                f"{sep}{col}{sep}", f"filtered_df['{col}']")

    exec_code = "filtered_df['ind_calculated']="+exec_code
    exec(exec_code)

    return filtered_df['ind_calculated']


def add_indicators_to_df(con: engine.Connection, df_in: pd.DataFrame, dts_name: str, symbol: str = None) -> pd.DataFrame:
    """calculates indicators series for a dataframe et returns the dataframe completed

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
        df_in (pd.DataFrame): Dataframe with all price data needed to calculate the indicators
        dts_name (str): Name of the dataset in the DB
        symbol (str,optional): Symbol for the Dataset, if None get symbol from df_in.CODE Default None

    Returns:
        pd.DataFrame: completed dataframe with indicators
    """
    df_comp = df_in.copy()
    if symbol==None:
        symbol=df_comp['CODE'][0]
    df_list_ind = get_ind_for_dts(
        con=con, dts_name=dts_name, symbol_code=symbol)
    for row in df_list_ind.itertuples(index=False):
        df_comp[row.LABEL] = get_indicator_value(
            df_in=df_comp, indic_code=row.PY_CODE)

    return df_comp


def drop_indicators_by_type(con: engine.Connection, df_in: pd.DataFrame, dts_name: str, symbol: str, ind_type: int = 0) -> pd.DataFrame:
    """drop indicators of a dataframe, get indicators list from the DB with the dataset name, symbol and indicator type 

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
        df_in (pd.DataFrame): Dataframe with column to drop
        dts_name (str): Name of the dataset in the DB
        symbol (str): Symbol for the Dataset
        ind_type (int, optional): type of indicator. Defaults to 0.

    Raises:
        ValueError: if no indicator found in DB for the inputs

    Returns:
        pd.DataFrame: a copy of the input dataframe without column dropped
    """
    list_ind = get_ind_list_by_type_for_dts(con, dts_name, symbol, ind_type)
    if len(list_ind) > 0:
        df_clean = df_in.copy()
        for lab in list_ind['LABEL'].tolist():
            df_clean.drop(lab, axis=1, inplace=True, errors='ignore')
    else:
        raise ValueError(
            f"no indicator found for dataset {dts_name}, symbol {symbol} and type {ind_type} ")

    return df_clean

def reorganize_columns(df: pd.DataFrame, column_order: str) -> pd.DataFrame:
    """
    Reorganize the column of a dataframe according to column_order

    Args:
        df (pd.DataFrame): input dataframe
        column_order (str): A string with the columns' names with the new sort "col1,col3,col4,col2".

    Returns:
        pd.DataFrame: the new dataframe

    Example:
        df = reorganize_columns(df, "col1,col3,col4,col2")
    """
    columns = [col.strip() for col in column_order.split(',')]
    if df.index.name in columns:
        columns.remove(df.index.name)

    return df[columns]

def drop_indicators_not_selected(con: engine.Connection, df_in: pd.DataFrame, dts_name: str, symbol: str, label: str, algo: str, organize: bool = False) -> pd.DataFrame:
    """drop useless indicators of a dataframe, get indicators list from the DB with the dataset name, symbol and label and algo

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
        df_in (pd.DataFrame): Dataframe with column to drop
        dts_name (str): Name of the dataset in the DB
        symbol (str): Symbol for the Dataset
        label (str) : name of the studied  label in the model
        algo (str) : type of algo of the model eg : RANDOM_FOREST_REG
        organize (bool) : If True, get the column list order from db and reorganize the df Default False

    Raises:
        ValueError: if no indicator found in DB for the inputs

    Returns:
        pd.DataFrame: a copy of the input dataframe without column dropped
    """
    mod_name = f"{symbol}_{dts_name}_{label}_{algo}"
    list_ind = get_ind_list_for_model(con, mod_name)
    df_clean = df_in.copy()
    if len(list_ind) > 0:
        list_ind=list_ind['LABEL'].tolist()
        list_ind.append(label)
        cols_to_drop = list(set(df_clean.columns)-set(list_ind))
        df_clean.drop(cols_to_drop,axis=1,inplace=True)

        if organize:
            list_col = get_header_for_model(con,mod_name)
            df_clean = reorganize_columns(df_clean, list_col)
    else:
        raise ValueError(f"no indicator found for model {mod_name}")

    return df_clean


if __name__ == "__main__":
    code = "ta.trend.SMAIndicator(close=$$CLOSE$$,window=20).sma_indicator()"
    con = get_connection()
    symb = "CW8"
    dts = "DCA_CLOSE_1D_V1"
    label="lab_perf_21d"
    algo="RANDOM_FOREST_REG"
    df = get_candles_to_df(con=con, symbol=symb, only_close=True)
    df = add_indicators_to_df(con=con, df_in=df, dts_name=dts)
    print(df[50:55])
    df = drop_indicators_by_type(con, df, dts, symb, 0)
    print(df[50:55])
    df=drop_indicators_not_selected(con, df, dts, symb,label,algo)
    print(df[50:55])
