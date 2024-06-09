import os
import re
from typing import Any, Dict, Optional, Type
import pandas as pd
from datetime import datetime as dt
from sqlalchemy import create_engine, engine, text, exc, pool, func
from sqlalchemy.orm import Session, sessionmaker, Query
from db_models import Base, Symbol, SymbolInfo

""" List of functions to import/export data from sqlite db
"""


def get_connection(str_db_path: str = None) -> engine.Connection:
    """ Create a new Connection instance to the marketdataenrich database
    Connection infos must be set in a .env file readable by decouple
    hard set port at 3306

    Args:
        str_db_name (str): the path+name of the db Default dataset_market

    Returns:
        Connection: The connection
    """

    if str_db_path == None:
        try:  # Try if we are in google colab or not
            import google.colab
            str_db_path = "/content/drive/MyDrive/COLAB/SQLITE_DB/dataset_market.db"
        except ImportError:
            str_db_path = "C:\Projets\Data\sqlite\dataset_market.db"

    str_db_path = str_db_path.replace("\\", "\\\\")
    conn_str = f"sqlite:///{str_db_path}"

    my_con = None
    try:
        my_con = create_engine(conn_str, poolclass=pool.NullPool).connect()
    except (exc.SQLAlchemyError) as e:
        print(f"Exception while opening connection {e} at {str_db_path}")

    return my_con


def close_connection(con: engine.Connection):
    """Close the connection pool

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
    """
    try:
        con.execute("SELECT 1")  # HINT TO AVOID THE BIG STACK
        con.close()
        con.engine.dispose()
    except (Exception, exc.SQLAlchemyError, exc.DBAPIError):
        pass


def get_model(session: sessionmaker, model: Type[Base], filter_values: Dict[str, Optional[Any]] = None) -> Query:
    """Get records in the database.

    Args:
        session (sessionmaker): SQLAlchemy session.
        model (Type[Base]): SQLAlchemy model class.
        filter_values(Dict) : Dict key/value to filter the table

    Returns:
        Query: SQLAlchemy query object with applied filters.
    """

    return session.query(model).filter_by(**filter_values).all()


def upsert_model(session: sessionmaker, model: Type[Base], primary_key_values: Dict[str, Optional[Any]] = None, **data) -> None:
    """Create or update a record in the database.

    Args:
        session (sessionmaker): SQLAlchemy session.
        model (Type[Base]): SQLAlchemy model class.
        primary_key_values(Dict) : Dict key/value to searh the line exists
    """
    if not primary_key_values:
        instance = False
    else:
        instance = session.query(model).filter_by(**primary_key_values).first()

    if instance:
        # print(f"update {primary_key_values}")
        for key, value in data.items():
            setattr(instance, key, value)
    else:
        # print(f"insert {data}")
        instance = model(**data)
        session.add(instance)
    session.commit()


def get_symbol(session: Session, symbol_code: str) -> Symbol:
    """returns data of the table SYMBOL for the symbol code

    Args:
        session (Session): SQLAlchemy session.
        symbol_code (str): the code of the symbol

    Returns:
        Symbol : The symbol object
    """
    try:
        res = session.query(Symbol).filter_by(code=symbol_code).first()
    except exc.SQLAlchemyError as e:
        print(f"Error fetching symbol for {symbol_code}: {e}")
        res = None
    return res


def get_list_symbol(session: Session, filter_values: Dict[str, Optional[Any]] = None) -> Query:
    """returns list of SYMBOL for the input filters

    Args:
        session (Session): SQLAlchemy session.
        filter_values(Dict) : Dict key/value to filter the table

    Returns:
        Query : The list of Symbol
    """
    try:
      return session.query(Symbol).filter_by(**filter_values).order_by(Symbol.sk_symbol).all()

    except exc.SQLAlchemyError as e:
        print(f"Error fetching symbol info for {filter_values}: {e}")
        return None


def upsert_symbol(**data) -> Symbol:
    """Add a symbol in the sqlite database if it doesn't exist yet

    Args:
        data (str): data for the symbol, need at least code

    Raises:
        ValueError: if there is no code in params
        DatabaseError: if the upsert fails

    Returns:
        Symbol : The upserted symbol object
    """
    symbol_code = None
    symbol = None

    if 'code' not in data:
        raise ValueError("Params must include code!")
    else:
        try:
            primary_key_column = 'code'
            primary_key_values = {
                key: value for key, value in data.items() if key in primary_key_column}
            symbol_code = primary_key_values[primary_key_column]
            # print(f"{symbol_code=} -- {primary_key_values=} {data=}")

            sym_con = get_connection()
            my_session_maker = sessionmaker(bind=sym_con)
            session = my_session_maker()
            upsert_model(session=session, model=Symbol,
                         primary_key_values=primary_key_values, **data)
            session.close()
            close_connection(sym_con)
        except (Exception, exc.SQLAlchemyError, exc.DBAPIError) as e:
            raise exc.DatabaseError(f"Error upserting Symbol Info {data}: {e}")

        symbol = get_symbol(symbol_code)
        if symbol == None:
            raise exc.DatabaseError(
                statement=f"ERROR upserting Symbol {data} !", params=data, orig=exc.DatabaseError)

    return symbol


def upsert_symbol_info(session: Session, **data) -> bool:
    """Add a symbol_info in the sqlite database

    Args:
        session (Session): SQLAlchemy session.
        data (str): data for the symbol info, need at least CODE or SK_SYMBOL and INFO

    Raises:
        ValueError: if there is no CODE, SK_SYMBOL, INFO in params
        DatabaseError: if the upsert fails

    Returns:
        bool : True if data has been inserted
    """
    key_code, key_symbol, key_info = 'code', 'sk_symbol', 'info'
    ret = False

    if (key_code not in data and key_symbol not in data) or (key_info not in data):
        raise ValueError(
            f"Params must include {key_code} or {key_symbol} and {key_info} {data=}!")

    else:
        if key_code in data and key_symbol not in data:  # need to get the SK
            symbol = get_symbol(session=session, symbol_code=data[key_code])
            if symbol == None:
                raise ValueError(f"Symbol {data[key_code]} not found!")
            else:
                sk_symbol = symbol.sk_symbol
        else:
            sk_symbol = data[key_symbol]

        try:
            params = {
                key_symbol: sk_symbol,
                key_info: data[key_info],
                'UPDATE_DATE': dt.now(),
                'ACTIVE_ROW': 1
            }

            upsert_model(session=session, model=SymbolInfo, **params)

            ret = True
        except (Exception, exc.SQLAlchemyError, exc.DBAPIError) as e:
            orig = e if hasattr(e, 'orig') else None
            raise exc.DatabaseError(
                statement=f"Error upserting Symbol Info {sk_symbol=}!", orig=orig, params=data)

    return ret


def load_yahoo_df_into_sql(session: Session, con: engine.Connection, df_yahoo: pd.DataFrame, symbol_code: str = "XMULTI",  timeframe: int = 1440, del_duplicate: bool = False, target_table: str = "candle") -> int:
    """load a dataframe of yahoo data into the input target_table table in DB with insert mode
    if no symbol_code, the dataframe must include the SK_SYMBOL

    Args:
        session (Session): SQLAlchemy Framework DB session.
        con (engine.Engine): SQLAlchemy connection to the DB candle
        df_yahoo (pd.DataFrame): the dataframe at yahoo format,open datetime as index
        symbol_code (str): the code of the symbol Default XMULTI
        timeframe (int): timeframe of the data (1D=1440) Default 1440
        del_duplicate (bool) : if after candle insert, duplicated canles must be deleted Default False
        target_table (str) : The name of the table in sqlite Default candle

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        int: nb lines
    """
    df_insert = df_yahoo.copy()
    if symbol_code != "XMULTI":
        symbol = get_symbol(session=session, symbol_code=symbol_code)

        if symbol == None:
            raise ValueError(f"Symbol {symbol_code} is not known !")
        else:
            df_insert['SK_SYMBOL'] = symbol.sk_symbol
            filter_query = f"AND a.SK_SYMBOL={symbol.sk_symbol}"

    df_insert.rename({'Adj Close': 'ADJ_CLOSE',
                     'Stock Splits': 'STOCK_SPLITS',
                      'Capital Gains': 'CAPITAL_GAINS'}, axis=1, inplace=True)
    df_insert['OPEN_DATETIME'] = df_insert.index
    df_insert['TIMEFRAME'] = timeframe

    res_ins = df_insert.to_sql(
        target_table, con=con, index=False, if_exists='append')

    if del_duplicate:
        query_clean = text(f"""DELETE FROM {target_table}
                                WHERE SK_CANDLE IN (
	                                SELECT a.SK_CANDLE FROM {target_table} a
	                                    INNER JOIN {target_table} b ON a.SK_SYMBOL=b.SK_SYMBOL
	                                    AND a.OPEN_DATETIME=b.OPEN_DATETIME AND a.TIMEFRAME=b.TIMEFRAME
                                        AND a.SK_CANDLE<b.SK_candle
                                        WHERE a.TIMEFRAME={timeframe} {filter_query}
                                )""")
        con.execute(query_clean)

    return res_ins


def get_last_candle_date(session: Session, con: engine.Connection, symbol_code: str, timeframe: int = 1440) -> pd.Timestamp:
    """ return the date of the last candle for this symbol and timeframe

    Args:
        session (Session): SQLAlchemy Framework DB session.
        con (engine.Connection): SQLAlchemy connection to the data DB 
        symbol_code (str): the code of the symbol
        timeframe timeframe of the data (1D=1440). Defaults to 1440.

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        pd.Timestamp: The date of the last candle for this symbol and timeframe
    """

    symbol = get_symbol(session, symbol_code)
    if symbol == None:
        raise ValueError(f"Symbol {symbol_code} is not known !")

    query = text(f"""SELECT MAX(OPEN_DATETIME) as LAST_DATE FROM CANDLE can
    WHERE can.SK_SYMBOL={symbol.SK_SYMBOL} AND can.TIMEFRAME={timeframe}    """)
    df = pd.read_sql_query(query, con)
    str_date = df["LAST_DATE"][0]
    dt_date = pd.to_datetime(str_date, format='%Y-%m-%d %H:%M:%S')
    return dt_date

def check_date_get_candles(date=None):
    """return the date in the good format for the DB

    Args:
        date (str or datetime, optional): the date to format YYYY-MM-DD or timestamp. Defaults to None.

    Returns:
        str: the date in YYYY-MM-DD or None
    """
    if date is None:
        return None
    if isinstance(date, dt):
        return date.strftime("%Y-%m-%d")
    if len(date) > 0:
        if re.match("\d{4}-\d{2}-\d{2}", date) is not None:
            return date
        else:
            raise ValueError(f"date {date} must be YYYY-MM-DD")
    return None

def get_candles_to_df(session: Session, con: engine.Connection, symbol_code: str = None, target_table: str = "CANDLE", timeframe: int = 1440, only_close: bool = False, tradable:bool = False, date_start=None, date_end=None) -> pd.DataFrame:
    """ select candles from DB to create a dataframe

    Args:
        session (Session): SQLAlchemy Framework DB session.
        con (engine.Connection): SQLAlchemy connection to the data DB 
        symbol_code (str, optional): the code of the symbol, if None select all codes Defaults to None
        target_table (str,optional): name of the table with the candles Defauls to CANDLE
        timeframe (int, optional): timeframe of the data (1D=1440) Defaults to 1440.
        only_close (bool, optional): True=only close column False=All columns. Defaults to False.
        tradable (bool, optional): True=only tradable symbols when no symbol specified False=All symbols. Defaults to False.
        date_start (str or datetime, optional): Start of the selection YYYY-MM-DD or timestamp. Defaults to None.
        date_end (str or datetime, optional): End of the selection YYYY-MM-DD or timestamp. Defaults to None.

    Raises:
        ValueError: if dates are not in the good format or if the symbol is unknown

    Returns:
        pd.DataFrame: DF with the data
    """
    cond_symbol = ""
    list_index_col = ["CODE","OPEN_DATETIME"]
    objects = "CODE,OPEN_DATETIME, "

    if symbol_code != None:
        symbol = get_symbol(session, symbol_code)
        if symbol == None:
            raise ValueError(f"Symbol {symbol_code} is not known !")
        cond_symbol = f"AND can.SK_SYMBOL ={symbol.sk_symbol} "
        list_index_col = "OPEN_DATETIME"
        objects = f"'{symbol_code}' AS CODE,OPEN_DATETIME, "

    # if isinstance(date_start, dt):
    #     date_start = date_start.strftime("%Y-%m-%d")

    # if isinstance(date_end, dt):
    #     date_end = date_end.strftime("%Y-%m-%d")

    if only_close:
        objects += "CLOSE"
    else:
        objects += "OPEN,HIGH,LOW,CLOSE,VOLUME"

    # pattern_date = re.compile("\d{4}-\d{2}-\d{2}")
    date_start = check_date_get_candles(date_start)
    date_end = check_date_get_candles(date_end)
    cond_date = ""
    if date_start is not None :
        # if pattern_date.match(date_start) is not None:
        cond_date += f" AND OPEN_DATETIME>='{date_start}'"
        # else:
            # raise ValueError(f"date_start {date_start} must be YYYY-MM-DD")

    if date_end is not None and len(date_end) > 0:
        # if pattern_date.match(date_end) is not None:
        cond_date += f" AND OPEN_DATETIME<='{date_end}'"
        # else:
            # raise ValueError(f"date_end {date_end} must be YYYY-MM-DD")
        
    if tradable:
        cond_symbol = "AND can.SK_SYMBOL IN (SELECT SK_SYMBOL FROM SYMBOL WHERE TRADABLE=1) "

    query = text(f"""SELECT {objects} FROM {target_table} can 
    WHERE can.TIMEFRAME={timeframe} {cond_symbol} {cond_date}    """)
    print(f"DEBUG: {query}")

    return pd.read_sql_query(query, con, index_col=list_index_col)


def delete_candles_symbol(con: engine.Connection, symbol_code: str) -> engine.cursor:
    """ Delete candles lines for the symbol in the DB

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol_code (str): the code of the symbol

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        engine.cursor: a SQLAlchemy cursor
    """

    symbol = get_symbol(symbol_code)
    if symbol == None:
        raise ValueError(f"Symbol {symbol_code} is not known !")

    del_st = text(
        f"DELETE FROM CANDLE WHERE SK_SYMBOL={symbol.SK_SYMBOL})")
    return con.execute(del_st)


def check_candles_last_months(con: engine.Connection, symbol_code: str, timeframe: int = 1440) -> pd.DataFrame:
    """ Return info about cnadles for a given symbol and timeframe
    Data returned : the month YYYY-MM, nb candles, min(date),max(date),min(close),max(close)

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol_code (str): the code of the symbol
        timeframe (int, optional): timeframe of the data (1D=1440) Defaults to 1440.

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        pd.DataFrame: DF with the data
    """

    symbol = get_symbol(symbol_code)
    if symbol == None:
        raise ValueError(f"Symbol {symbol_code} is not known !")

    query = text(f""" SELECT  STRFTIME('%Y-%m',OPEN_DATETIME) AS MONTH ,COUNT(*) AS NB,
                MIN(OPEN_DATETIME),MAX(OPEN_DATETIME),MIN(CLOSE),MAX(CLOSE)   FROM CANDLE 
                WHERE SK_SYMBOL={symbol.SK_SYMBOL} AND TIMEFRAME={timeframe} 
                AND OPEN_DATETIME>DATE('now','start of month','-4 month')
                GROUP BY 1 ORDER BY 1 DESC""")
    return pd.read_sql_query(query, con, index_col='MONTH')


def get_info_all_stock(con: engine.Connection) -> pd.DataFrame:
    """ returns the basic information for all stocks

    Args:    
        con (engine.Connection): SQLAlchemy connection to the DB 
    Returns:
        pd.DataFrame: a dataframe  with stocks data : SK_SYMBOL,CODE,  NAME, TYPE , REGION, CODE_YAHOO, SHARESOUTSTANDING
    """

    query = text("""select s.sk_symbol ,s.code,s.name,s.type ,s.region,s.code_yahoo,s.tradable,si.sharesoutstanding 
            from SYMBOL s
            left join SYMBOL_INFO si on s.sk_symbol =si.sk_symbol and si.ACTIVE_ROW =1 
            where s.ACTIVE =1
            and s.type ='Stock' and s.sk_symbol <>417 order by s.sk_symbol 
              """)
    return pd.read_sql_query(query, con)

def get_ind_for_dts(con: engine.Connection, dts_name: str, symbol_code: str) -> pd.DataFrame:
    """ returns the indicators data in a dataframe for a given dataset and a symbol

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        dts_name (str): name of the dataset
        symbol_code (str): the code of the symbol

    Returns:
        pd.DataFrame: a dataframe  with indicators data : NAME LABEL PY_CODE
    """

    query = text(f"""SELECT ind.NAME,ind.LABEL,ind.CODE as PY_CODE FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='{dts_name}' AND sym.CODE='{symbol_code}' and ind.CODE is not null ORDER BY ind.SK_INDICATOR
    """)
    return pd.read_sql_query(query, con)




def get_ind_list_by_type_for_dts(con: engine.Connection, dts_name: str, symbol_code: str, ind_type: int = 0) -> pd.DataFrame:
    """ returns the list of labels for indicators for a given dataset, a symbol  and an indicator type

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        dts_name (str): name of the dataset
        symbol_code (str): the code of the symbol
        ind_type (int) : the type of indicator (0 intermediate 1 feature 2 label)

    Returns:
        pd.DataFrame: a dataframe  with indicators data :  LABEL 
    """
    query = text(f"""SELECT distinct ind.LABEL FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='{dts_name}' AND sym.CODE='{symbol_code}' and ind.TYPE={ind_type}""")
    return pd.read_sql_query(query, con)


def get_ind_list_for_model(con: engine.Connection, model_name: str) -> pd.DataFrame:
    """ returns the list of labels for filtered indicators for a given model

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        model_name (str): name of the model

    Returns:
        pd.DataFrame: a dataframe  with indicators data :  LABEL 
    """
    query = text(f"""SELECT distinct ind.LABEL  FROM model md
    INNER JOIN ds_filtered df ON md.SK_MODEL=df.SK_MODEL
    INNER JOIN indicator ind ON df.SK_INDICATOR=ind.SK_INDICATOR
    WHERE md.NAME='{model_name}'""")
    return pd.read_sql_query(query, con)


def get_header_for_model(con: engine.Connection, model_name: str) -> str:
    """ returns the list of features ordered for a given model

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        model_name (str): name of the model

    Returns:
        str: the list of features as a string "col1,col3,col4,col2" 
    """
    query = text(
        f"""SELECT distinct md.HEADER_DTS  FROM model md WHERE md.NAME='{model_name}' LIMIT 1""")
    df = pd.read_sql_query(query, con)
    return df["HEADER_DTS"][0]


if __name__ == "__main__":
    symbol = "OVH.PA"
    model_name = "CW8_DCA_CLOSE_1D_V1_lab_perf_21d_LSTM_CLASS"
    timeframe = 1440
    db_name = "dataset_market.db"
    candle_name = "candle_CW8.db"
    con_CW8 = get_connection("C:\Projets\Data\sqlite\candle_CW8.db")

    con_fwk = get_connection(
        str_db_path="C:\Projets\Data\sqlite\dataset_market.db")
    my_session_maker = sessionmaker(bind=con_fwk)
    session = my_session_maker()

    sym = get_symbol(session=session, symbol_code=symbol)
    # sym = create_symbol(symbol)

    # sym2=upsert_symbol(CODE='ABC',NAME='TEST BDU T',TYPE='Stock',CURRENCY='USD')

    print(f"SK du symbol {symbol} : {sym.sk_symbol} {sym.name=}")

    # my_filter={'SK_SYMBOL':308}
    # my_res=get_model(session=session,model=SymbolInfo,filter_values=my_filter)

    # my_filter = {"active": 1, "region": "France", "type": "Stock"}
    # my_res = get_list_symbol(session=session, filter_values=my_filter)
    # # print(f"{my_res=}")
    # for res in my_res:
    #     print(f"{res.sk_symbol=}  {res.name=} ")
    df_stocks=get_info_all_stock(con_fwk)
    print(df_stocks.head())

    # df_test = get_candles_to_df(session=session,
    #     con=con_CW8,symbol_code='CW8', date_start=dt.strptime('2023-01-01', '%Y-%m-%d'))
    # print(df_test.shape)

    # last_date = get_last_candle_date(
    #     con=con_CW8, symbol=symbol, timeframe=1440)
    # print(f"my last_date {last_date}")

    # print(check_candles_last_months(con_CW8, symbol, timeframe))
