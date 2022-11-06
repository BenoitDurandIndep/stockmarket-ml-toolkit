import os
import re
from decouple import AutoConfig
import pandas as pd
from sqlalchemy import create_engine, engine, text

""" List of functions to import/export data from maria db
"""


def get_connection() -> engine.Connection:
    """ Create a new Connection instance to the marketdataenrich database
    Connection infos must be set in a .env file readable by decouple
    hard set port at 3306

    Returns:
        Connection: The connection
    """
    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config = AutoConfig(search_path=os.path.join(dir_path, ".env"))
    MARIA_SERVER = config("MARIA_SERVER", cast=str)
    MARIA_DB = config("MARIA_DB", cast=str)
    MARIA_USER = config("MARIA_USER", cast=str)
    MARIA_PWD = config("MARIA_PWD", cast=str)
    conn_str = f"mysql+pymysql://{MARIA_USER}:{MARIA_PWD}@{MARIA_SERVER}:3306/{MARIA_DB}"

    return create_engine(conn_str).connect()

def close_connection(con: engine.Connection) :
    """Close the connection pool

    Args:
        con (engine.Connection): QLAlchemy connection to the DB
    """
    try:
        con.close()
        con.engine.dispose()
    except:
        pass


def get_symbol_info(con: engine.Connection, symbol: str) -> pd.DataFrame:
    """returns data of the table SYMBOL for the symbol code

    Args:
        con (engine.Engine): SQLAlchemy connection to the DB
        symbol (str): the code of the symbol

    Returns:
        pd.DataFrame: data in db for this symbol
    """
    query = f"SELECT SK_SYMBOL, CODE, NAME, TYPE, REGION, CURRENCY, COMMENT, CODE_ALPHA, CODE_YAHOO, CODE_ISIN FROM SYMBOL WHERE CODE='{symbol}'"
    return pd.read_sql_query(query, con, index_col='SK_SYMBOL')


def load_yahoo_df_into_sql(con: engine.Connection, df_yahoo: pd.DataFrame, symbol: str, timeframe: int) -> int:
    """load a dataframe of yahoo data into the CANDLE table in DB

    Args:
        con (engine.Engine): SQLAlchemy connection to the DB 
        df_yahoo (pd.DataFrame): the dataframe at yahoo format
        symbol (str): the code of the symbol
        timeframe (int): timeframe of the data (1D=1440)

    Raises:
        ValueError: if the symlbol is unknown

    Returns:
        int: nb lines
    """
    df_symbol = get_symbol_info(con, symbol)
    if df_symbol.empty:
        raise ValueError(f"Symbol {symbol} is not known !")

    df_insert = df_yahoo.copy()
    df_insert.rename({'Adj Close': 'ADJ_CLOSE'}, axis=1, inplace=True)
    df_insert['OPEN_DATETIME'] = df_insert.index
    df_insert['SK_SYMBOL'] = df_symbol.index[0]
    df_insert['TIMEFRAME'] = timeframe

    return df_insert.to_sql("candle", con=con, index=False, if_exists='append')


def get_candles_to_df(con: engine.Connection, symbol: str, timeframe: int = 1440, only_close: bool = False, date_start: str = None, date_end: str = None) -> pd.DataFrame:
    """ select candles from DB to create a dataframe

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol (str): the code of the symbol
        timeframe (int, optional): timeframe of the data (1D=1440) Defaults to 1440.
        only_close (bool, optional): True=only close column False=All columns. Defaults to False.
        date_start (str, optional): Start of the selection YYYY-MM-DD. Defaults to None.
        date_end (str, optional): End of the selection YYYY-MM-DD. Defaults to None.

    Raises:
        ValueError: if dates are not in the good format

    Returns:
        pd.DataFrame: DF with the data
    """
    objects = "CODE,OPEN_DATETIME, "
    if only_close:
        objects += "CLOSE"
    else:
        objects += "OPEN,HIGH,LOW,CLOSE,ADJ_CLOSE,VOLUME"

    pattern_date = re.compile("\d{4}-\d{2}-\d{2}")
    cond_date = ""
    if date_start is not None and len(date_start) > 0:
        if pattern_date.match(date_start) is not None:
            cond_date += f" AND OPEN_DATETIME>='{date_start}'"
        else:
            raise ValueError(f"date_start {date_start} must be YYYY-MM-DD")

    if date_end is not None and len(date_end) > 0:
        if pattern_date.match(date_end) is not None:
            cond_date += f" AND OPEN_DATETIME<='{date_end}'"
        else:
            raise ValueError(f"date_end {date_end} must be YYYY-MM-DD")

    query = f"""SELECT {objects} FROM SYMBOL sym INNER JOIN CANDLE can ON sym.SK_SYMBOL=can.SK_SYMBOL 
    WHERE sym.CODE='{symbol}' AND can.TIMEFRAME={timeframe} {cond_date}
    """
    #print(query)
    return pd.read_sql_query(query, con, index_col='OPEN_DATETIME')


def delete_candles_symbol(con: engine.Connection, symbol: str) -> engine.cursor:
    """ Delete candles lines for the symbol in the DB

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol (str): the code of the symbol

    Returns:
        engine.cursor: a SQLAlchemy cursor
    """
    del_st = text(
        f"DELETE FROM CANDLE WHERE SK_SYMBOL IN (SELECT SK_SYMBOL FROM SYMBOL WHERE CODE='{symbol}')")
    res = con.execute(del_st)
    return res

def get_ind_for_dts(con: engine.Connection, dts_name: str,symbol: str)->pd.DataFrame:
    """ returns the indicators data in a dataframe for a given dataset and a symbol

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        dts_name (str): name of the dataset
        symbol (str): the code of the symbol

    Returns:
        pd.DataFrame: a dataframe  with indicators data : NAME LABEL PY_CODE
    """    

    query = f"""SELECT ind.NAME,ind.LABEL,ind.CODE as PY_CODE FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='{dts_name}' AND sym.CODE='{symbol}' and ind.CODE is not null
    """
    return pd.read_sql_query(query, con)


if __name__ == "__main__":
    symbol = "CW8"
    con = get_connection()
    sym = get_symbol_info(con, symbol)
    print(f"SK du symbol {symbol} : {sym.index[0]}")

    df_test = get_candles_to_df(con=con, symbol=symbol, date_end='2021-01-01')
    print(df_test.shape)
