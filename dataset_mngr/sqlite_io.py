import os
import re
from decouple import AutoConfig
import pandas as pd
from datetime import datetime as dt
from sqlalchemy import create_engine, engine, text, exc,pool

""" List of functions to import/export data from maria db
"""


def get_conf(name: str, file_name: str = ".env", dir_path:str=None) -> str:
    """return the value of a conf stored in file using AutoConfig

    Args:
        name (str): name of the key
        file_name (str, optional): name of the file. Defaults to ".env".
        dir_path (str, optional) : path of the file_name. If None in the .env directory

    Returns:
        str: the value as a string
    """
    if dir_path==None:
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    
    config = AutoConfig(search_path=os.path.join(dir_path, file_name))

    return config(name, cast=str)


def get_connection(str_db_path:str=None) -> engine.Connection:
    """ Create a new Connection instance to the marketdataenrich database
    Connection infos must be set in a .env file readable by decouple
    hard set port at 3306

    Args:
        str_db_name (str): the path+name of the db Default dataset_market

    Returns:
        Connection: The connection
    """

    if str_db_path==None:
        str_db_path = get_conf("SQLITE_DB_FWK")

    str_db_path=str_db_path.replace("\\","\\\\")
    conn_str = f"sqlite:///{str_db_path}"

    try:
        my_con=create_engine(conn_str, poolclass=pool.NullPool).connect()
    except(exc.SQLAlchemyError) as e:
        print(f"Exception while opening connection {e}")

    return my_con


def close_connection(con: engine.Connection):
    """Close the connection pool

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
    """
    try:
        con.execute("SELECT 1") #HINT TO AVOID THE BIG STACK
        con.close()
        con.engine.dispose()
    except (Exception,exc.SQLAlchemyError,exc.DBAPIError) as e :
        print(f"Exception while closing {e}")


def get_symbol_info(symbol: str) -> pd.DataFrame:
    """returns data of the table SYMBOL for the symbol code

    Args:
        symbol (str): the code of the symbol

    Returns:
        pd.DataFrame: data in db for this symbol
    """
    query = text(
        f"SELECT SK_SYMBOL, CODE, NAME, TYPE, REGION, CURRENCY, COMMENT, CODE_ALPHA, CODE_YAHOO, CODE_ISIN FROM SYMBOL WHERE CODE='{symbol}'")

    sym_con=get_connection()
    df_res=pd.read_sql_query(query, sym_con, index_col='SK_SYMBOL')
    close_connection(sym_con)

    return df_res


def load_yahoo_df_into_sql(con: engine.Connection, df_yahoo: pd.DataFrame, symbol: str, timeframe: int, del_duplicate: bool = False) -> int:
    """load a dataframe of yahoo data into the CANDLE table in DB with insert mode

    Args:
        con (engine.Engine): SQLAlchemy connection to the DB 
        df_yahoo (pd.DataFrame): the dataframe at yahoo format
        symbol (str): the code of the symbol
        timeframe (int): timeframe of the data (1D=1440)
        del_duplicate (bool) : if after candle insert, duplicated canles must be deleted Default False

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        int: nb lines
    """
    df_symbol = get_symbol_info( symbol)
    if df_symbol.empty:
        raise ValueError(f"Symbol {symbol} is not known !")

    df_insert = df_yahoo.copy()
    df_insert.rename({'Adj Close': 'ADJ_CLOSE',
                     'Stock Splits': 'STOCK_SPLITS'}, axis=1, inplace=True)
    df_insert['OPEN_DATETIME'] = df_insert.index
    df_insert['SK_SYMBOL'] = df_symbol.index[0]
    df_insert['TIMEFRAME'] = timeframe

    res_ins = df_insert.to_sql(
        "candle", con=con, index=False, if_exists='append')

    if del_duplicate:
        query_clean = text(f"""DELETE FROM CANDLE
                                WHERE SK_CANDLE IN (
	                                SELECT a.SK_CANDLE FROM CANDLE a
	                                    INNER JOIN CANDLE b ON a.SK_SYMBOL=b.SK_SYMBOL
	                                    AND a.OPEN_DATETIME=b.OPEN_DATETIME AND a.TIMEFRAME=b.TIMEFRAME
                                        AND a.SK_CANDLE<b.SK_candle
                                        WHERE a.SK_SYMBOL={df_symbol.index[0]} AND a.TIMEFRAME={timeframe}
                                )""")
        con.execute(query_clean)

    return res_ins


def get_last_candle_date(con: engine.Connection, symbol: str, timeframe: int = 1440) -> pd.Timestamp:
    """ return the date of the last candle for this symbol and timeframe

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol (str): the code of the symbol
        timeframe timeframe of the data (1D=1440). Defaults to 1440.

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        pd.Timestamp: The date of the last candle for this symbol and timeframe
    """

    df_symbol = get_symbol_info( symbol)
    if df_symbol.empty:
        raise ValueError(f"Symbol {symbol} is not known !")
    
    query = text(f"""SELECT MAX(OPEN_DATETIME) as LAST_DATE FROM CANDLE can
    WHERE can.SK_SYMBOL={df_symbol.index[0]} AND can.TIMEFRAME={timeframe}    """)
    df = pd.read_sql_query(query, con)
    str_date=df["LAST_DATE"][0]
    dt_date=pd.to_datetime(str_date,format= '%Y-%m-%d %H:%M:%S')
    return dt_date


def get_candles_to_df(con: engine.Connection, symbol: str, timeframe: int = 1440, only_close: bool = False, date_start= None, date_end= None) -> pd.DataFrame:
    """ select candles from DB to create a dataframe

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol (str): the code of the symbol
        timeframe (int, optional): timeframe of the data (1D=1440) Defaults to 1440.
        only_close (bool, optional): True=only close column False=All columns. Defaults to False.
        date_start (str or datetime, optional): Start of the selection YYYY-MM-DD or timestamp. Defaults to None.
        date_end (str or datetime, optional): End of the selection YYYY-MM-DD or timestamp. Defaults to None.


    Raises:
        ValueError: if dates are not in the good format or if the symbol is unknown

    Returns:
        pd.DataFrame: DF with the data
    """

    df_symbol = get_symbol_info( symbol)
    if df_symbol.empty:
        raise ValueError(f"Symbol {symbol} is not known !")

    if isinstance(date_start, dt):
        date_start = date_start.strftime("%Y-%m-%d")

    if isinstance(date_end, dt):
        date_end = date_end.strftime("%Y-%m-%d")

    objects = f"'{symbol}' AS CODE,OPEN_DATETIME, "
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

    query = text(f"""SELECT {objects} FROM CANDLE can 
    WHERE can.SK_SYMBOL ={df_symbol.index[0]} AND can.TIMEFRAME={timeframe} {cond_date}    """)
    # print(query)
    return pd.read_sql_query(query, con, index_col='OPEN_DATETIME')


def delete_candles_symbol(con: engine.Connection, symbol: str) -> engine.cursor:
    """ Delete candles lines for the symbol in the DB

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol (str): the code of the symbol

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        engine.cursor: a SQLAlchemy cursor
    """

    df_symbol = get_symbol_info( symbol)
    if df_symbol.empty:
        raise ValueError(f"Symbol {symbol} is not known !")
    
    del_st = text(
        f"DELETE FROM CANDLE WHERE SK_SYMBOL={df_symbol.index[0]})")
    return con.execute(del_st)


def check_candles_last_months(con: engine.Connection, symbol: str, timeframe: int = 1440) -> pd.DataFrame:
    """ Return info about cnadles for a given symbol and timeframe
    Data returned : the month YYYY-MM, nb candles, min(date),max(date),min(close),max(close)

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol (str): the code of the symbol
        timeframe (int, optional): timeframe of the data (1D=1440) Defaults to 1440.

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        pd.DataFrame: DF with the data
    """

    df_symbol = get_symbol_info( symbol)
    if df_symbol.empty:
        raise ValueError(f"Symbol {symbol} is not known !")
    
    query = text(f""" SELECT  STRFTIME('%Y-%m',OPEN_DATETIME) AS MONTH ,COUNT(*) AS NB,
                MIN(OPEN_DATETIME),MAX(OPEN_DATETIME),MIN(CLOSE),MAX(CLOSE)   FROM CANDLE 
                WHERE SK_SYMBOL={df_symbol.index[0]} AND TIMEFRAME={timeframe} 
                AND OPEN_DATETIME>DATE('now','start of month','-4 month')
                GROUP BY 1 ORDER BY 1 DESC""")
    return pd.read_sql_query(query, con, index_col='MONTH')


def get_ind_for_dts(con: engine.Connection, dts_name: str, symbol: str) -> pd.DataFrame:
    """ returns the indicators data in a dataframe for a given dataset and a symbol

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        dts_name (str): name of the dataset
        symbol (str): the code of the symbol

    Returns:
        pd.DataFrame: a dataframe  with indicators data : NAME LABEL PY_CODE
    """

    query = text(f"""SELECT ind.NAME,ind.LABEL,ind.CODE as PY_CODE FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='{dts_name}' AND sym.CODE='{symbol}' and ind.CODE is not null ORDER BY ind.SK_INDICATOR
    """)
    return pd.read_sql_query(query, con)


def get_ind_list_by_type_for_dts(con: engine.Connection, dts_name: str, symbol: str, ind_type: int = 0) -> pd.DataFrame:
    """ returns the list of labels for indicators for a given dataset, a symbol  and an indicator type

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        dts_name (str): name of the dataset
        symbol (str): the code of the symbol
        ind_type (int) : the type of indicator (0 intermediate 1 feature 2 label)

    Returns:
        pd.DataFrame: a dataframe  with indicators data :  LABEL 
    """
    query = f"""SELECT distinct ind.LABEL FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='{dts_name}' AND sym.CODE='{symbol}' and ind.TYPE={ind_type}"""
    return pd.read_sql_query(query, con)


def get_ind_list_for_model(con: engine.Connection, model_name: str) -> pd.DataFrame:
    """ returns the list of labels for filtered indicators for a given model

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        model_name (str): name of the model

    Returns:
        pd.DataFrame: a dataframe  with indicators data :  LABEL 
    """
    query = f"""SELECT distinct ind.LABEL  FROM model md
    INNER JOIN ds_filtered df ON md.SK_MODEL=df.SK_MODEL
    INNER JOIN indicator ind ON df.SK_INDICATOR=ind.SK_INDICATOR
    WHERE md.NAME='{model_name}'"""
    return pd.read_sql_query(query, con)


def get_header_for_model(con: engine.Connection, model_name: str) -> str:
    """ returns the list of features ordered for a given model

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        model_name (str): name of the model

    Returns:
        str: the list of features as a string "col1,col3,col4,col2" 
    """
    query = f"""SELECT distinct md.HEADER_DTS  FROM model md WHERE md.NAME='{model_name}' LIMIT 1"""
    df = pd.read_sql_query(query, con)
    return df["HEADER_DTS"][0]


if __name__ == "__main__":
    symbol = "CW8"
    model_name = "CW8_DCA_CLOSE_1D_V1_lab_perf_21d_LSTM_CLASS"
    timeframe = 1440
    db_name="dataset_market.db"
    candle_name="candle_CW8.db"
    con_CW8 = get_connection("C:\Projets\Data\sqlite\candle_CW8.db")

    sym = get_symbol_info( symbol)

    print(f"SK du symbol {symbol} : {sym.index[0]}")

    df_test = get_candles_to_df(con=con_CW8, symbol=symbol, date_start=dt.strptime('2023-01-01','%Y-%m-%d'))
    print(df_test.shape)

    last_date = get_last_candle_date(con=con_CW8, symbol=symbol, timeframe=1440)
    print(f"my last_date {last_date}")

    print(check_candles_last_months(con_CW8, symbol, timeframe))
