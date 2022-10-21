import pandas as pd
import ta
from sqlalchemy import  engine
from maria_import_export import get_connection, get_candles_to_df, get_ind_for_dts

KEY_WORDS_LIST = ["CLOSE", "OPEN", "HIGH", "LOW", "IND"]
DEFAULT_INDIC_SEP="$$"


def get_indicator_value(df_in: pd.DataFrame, indic_code: str, sep: str = DEFAULT_INDIC_SEP) -> pd.Series:
    """ calculate and return an indicator serie 
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
    #print(tab_ind)
    for col in tab_ind:
        if col in KEY_WORDS_LIST:
            if col=="IND" : # case indicator based onan other indicator
                for code in tab_ind:
                    if code.startswith("!"):
                        code=code[1:]
                        filtered_df[code] = df_in[code]
                        exec_code = exec_code.replace(
                            f"{sep}{col}{sep}!{code}{sep}{col}{sep}", f"filtered_df['{code}']")
                        
            else :
                filtered_df[col] = df_in[col]
                exec_code = exec_code.replace(
                    f"{sep}{col}{sep}", f"filtered_df['{col}']")

    exec_code = "filtered_df['ind_calculated']="+exec_code
    #print(exec_code)
    exec(exec_code)
    return filtered_df['ind_calculated']


def add_indicators(con: engine.Connection,df_in: pd.DataFrame, dts_name: str) -> pd.DataFrame:
    """calculates indicators series for a dataframe et returns the dataframe completed

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
        df_in (pd.DataFrame): Dataframe with all price data needed to calculate the indicators
        dts_name (str): Name o ffthe dataset in the DB

    Returns:
        pd.DataFrame: completed dataframe with indicators
    """    
    df_comp = df_in.copy()
    df_list_ind=get_ind_for_dts(con=con,dts_name=dts_name,symbol=df_comp['CODE'][0])
    for row in df_list_ind.itertuples(index=False):
        df_comp[row.LABEL]=get_indicator_value(df_in=df_comp, indic_code=row.PY_CODE)

    return df_comp


if __name__ == "__main__":
    code = "ta.trend.SMAIndicator(close=$$CLOSE$$,window=20).sma_indicator()"
    con = get_connection()
    symb = "CW8"
    dts="DCA_CLOSE_1D_V1"
    df = get_candles_to_df(con=con, symbol=symb, only_close=True)
    #df['SMA20'] = get_indicator_value(df_in=df, indic_code=code)
    df=add_indicators(con=con,df_in=df,dts_name=dts)
    print(df[50:55])
