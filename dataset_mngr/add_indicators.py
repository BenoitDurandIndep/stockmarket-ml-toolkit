import pandas as pd
import numpy as np
import time
import ta

from maria_import_export import get_connection,get_candles_to_df

KEY_WORDS_LIST = ["CLOSE", "OPEN", "HIGH", "LOW", "IND"]


def get_indicator_value(df_in: pd.DataFrame, indic_code: str, sep: str = "$$") -> pd.Series:
    """ calculate and return an indicator serie 
    Column used in the code must be surounded by sep

    Example of indic_code : ta.trend.SMAIndicator(close=$$CLOSE$$,window=).sma_indicator()

    Args:
        df_in (pd.DataFrame): Dataframe with the data needed for calculation
        indic_code (str): the code of the indicator 
        sep (str, optional): sep to split key words in the code. Defaults to "$$".

    Returns:
        pd.Series: the calculated indicator
    """

    filtered_df = pd.DataFrame()
    ind_val = pd.Series(dtype='float64')
    exec_code = indic_code
    tab_ind = indic_code.split(sep)

    for col in tab_ind:
        if col in KEY_WORDS_LIST:
            filtered_df[col] = df_in[col]
            exec_code = indic_code.replace(
                f"{sep}{col}{sep}", f"filtered_df[{col}]")

    exec_code = "ind_val="+exec_code
    print(exec_code)
    exec(exec_code)
    return ind_val


if __name__ == "__main__":
    code = "ta.trend.SMAIndicator(close=$$CLOSE$$,window=20).sma_indicator()"
    con = get_connection()
    symb = "CW8"
    df = get_candles_to_df(con=con, symbol=symb, only_close=True)
    ser = get_indicator_value(df, code)
    ser.describe()
