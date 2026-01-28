from decimal import ROUND_HALF_UP, Decimal
import json
import os
import re
from typing import Any, Dict, Optional, Type, List, TYPE_CHECKING, TypeVar
import numpy as np
import pandas as pd
from datetime import datetime as dt
from sqlalchemy import create_engine, engine, text, exc, pool, func
from sqlalchemy.engine import Connection, CursorResult
from sqlalchemy.orm import Session, sessionmaker, Query
from sqlalchemy.exc import SQLAlchemyError
from .db_models import Base, Campaign, Symbol, SymbolInfo, CombiModels, BtResult

if TYPE_CHECKING:
    from backtest.backtest_preparation import Strategy

T = TypeVar("T", bound=Base)

""" List of functions to import/export data from sqlite db
"""


def get_connection(str_db_path: Optional[str] = None) -> Optional[Connection]:
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
            import google.colab  # type: ignore
            str_db_path = "/content/drive/MyDrive/COLAB/SQLITE_DB/dataset_market.db"
        except ImportError:
            str_db_path = r"D:\Projets\Data\sqlite\dataset_market.db"

    str_db_path = str_db_path.replace("\\", "\\\\")
    conn_str = f"sqlite:///{str_db_path}"

    my_con: Optional[Connection] = None
    try:
        my_con = create_engine(conn_str, poolclass=pool.NullPool).connect()
    except (exc.SQLAlchemyError) as e:
        print(f"Exception while opening connection {e} at {str_db_path}")

    return my_con


def close_connection(con: Connection):
    """Close the connection pool

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB
    """
    try:
        con.execute(text("SELECT 1"))  # HINT TO AVOID THE BIG STACK
        con.close()
        con.engine.dispose()
    except (Exception, exc.SQLAlchemyError, exc.DBAPIError):
        pass


def get_model(session: Session, model: Type[Base], filter_values: Optional[Dict[str, Optional[Any]]] = None) -> List[Base]:
    """Get records in the database.

    Args:
        session (sessionmaker): SQLAlchemy session.
        model (Type[Base]): SQLAlchemy model class.
        filter_values(Dict) : Dict key/value to filter the table

    Returns:
        Query: SQLAlchemy query object with applied filters.
    """

    filter_values = filter_values or {}
    return session.query(model).filter_by(**filter_values).all()


def upsert_model(session: Session, model: Type[Base], primary_key_values: Optional[Dict[str, Optional[Any]]] = None, **data) -> None:
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


def get_symbol(session: Session, symbol_code: str) -> Optional[Symbol]:
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


def get_list_symbol(session: Session, filter_values: Optional[Dict[str, Optional[Any]]] = None) -> List[Symbol]:
    """returns list of SYMBOL for the input filters

    Args:
        session (Session): SQLAlchemy session.
        filter_values(Dict) : Dict key/value to filter the table

    Returns:
        Query : The list of Symbol
    """
    try:
        filter_values = filter_values or {}
        return session.query(Symbol).filter_by(**filter_values).order_by(Symbol.sk_symbol).all()
    except exc.SQLAlchemyError as e:
        print(f"Error fetching symbol info for {filter_values}: {e}")
        return []


def upsert_symbol(**data) -> Optional[Symbol]:
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
            symbol = get_symbol(session=session, symbol_code=symbol_code)
            session.close()
            if sym_con is not None:
                close_connection(sym_con)
        except (Exception, exc.SQLAlchemyError, exc.DBAPIError) as e:
            raise exc.DatabaseError(statement=f"Error upserting Symbol Info {data}", params=data, orig=e)
        if symbol == None:
            raise exc.DatabaseError(
                statement=f"ERROR upserting Symbol {data} !", params=data, orig=ValueError("Symbol not found after upsert"))

    return symbol


def _resolve_sk_symbol(session: Session, data: Dict[str, Any]) -> int:
    key_code, key_symbol = "code", "sk_symbol"
    if key_symbol in data:
        return data[key_symbol]
    symbol = get_symbol(session=session, symbol_code=data[key_code])
    if symbol is None:
        raise ValueError(f"Symbol {data[key_code]} not found!")
    return symbol.sk_symbol


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

    try:
        sk_symbol = _resolve_sk_symbol(session, data)
        params = {
            key_symbol: sk_symbol,
            key_info: data[key_info],
            "UPDATE_DATE": dt.now(),
            "ACTIVE_ROW": 1,
        }
        upsert_model(session=session, model=SymbolInfo, **params)
        ret = True
    except (Exception, exc.SQLAlchemyError, exc.DBAPIError) as e:
        raise exc.DatabaseError(
            statement=f"Error upserting Symbol Info {sk_symbol=}!", orig=e, params=data)

    return ret


def load_yahoo_df_into_sql(session: Session, con: Connection, df_yahoo: pd.DataFrame, symbol_code: str = "XMULTI",  timeframe: int = 1440, del_duplicate: bool = False, target_table: str = "candle") -> Optional[int]:
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


def get_last_candle_date(session: Session, con: Connection, symbol_code: str, timeframe: int = 1440) -> pd.Timestamp:
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
    WHERE can.SK_SYMBOL={symbol.sk_symbol} AND can.TIMEFRAME={timeframe}    """)
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
        if re.match("\\d{4}-\\d{2}-\\d{2}", date) is not None:
            return date
        else:
            raise ValueError(f"date {date} must be YYYY-MM-DD")
    return None

def get_candles_to_df(session: Session, con: Connection, symbol_code: Optional[str] = None, target_table: str = "CANDLE", timeframe: int = 1440,
                       only_close: bool = False,columns_name: str="OPEN,HIGH,LOW,CLOSE,VOLUME", tradable:bool = False, skip_cond:bool = False, date_start=None, date_end=None) -> pd.DataFrame:
    """ select candles from DB to create a dataframe

    Args:
        session (Session): SQLAlchemy Framework DB session.
        con (engine.Connection): SQLAlchemy connection to the data DB 
        symbol_code (str, optional): the code of the symbol, if None select all codes Defaults to None
        target_table (str,optional): name of the table with the candles Defauls to CANDLE
        timeframe (int, optional): timeframe of the data (1D=1440) Defaults to 1440.
        only_close (bool, optional): True=only close column False=All columns. Defaults to False.
        columns_name (str, optional): the columns to select Defaults to "OPEN,HIGH,LOW,CLOSE,VOLUME"
        tradable (bool, optional): True=only tradable symbols when no symbol specified False=All symbols. Defaults to False.
        skip_cond (bool, optional): True=skip the conditions False=use the symbol and timeframe condition. Defaults to False.
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

    if only_close:
        objects += "CLOSE"
    else:
        objects += columns_name

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

    str_query = f"SELECT {objects} FROM {target_table} can"
    if skip_cond==False:
        str_query = f"{str_query} WHERE can.TIMEFRAME={timeframe} {cond_symbol} {cond_date} "
    query=text(str_query)
    print(f"DEBUG: {query}")

    return pd.read_sql_query(query, con, index_col=list_index_col)


def delete_candles_symbol(session: Session, con: Connection, symbol_code: str) -> CursorResult:
    """ Delete candles lines for the symbol in the DB

    Args:
        con (engine.Connection): SQLAlchemy connection to the DB 
        symbol_code (str): the code of the symbol

    Raises:
        ValueError: if the symbol is unknown

    Returns:
        engine.cursor: a SQLAlchemy cursor
    """

    symbol = get_symbol(session=session, symbol_code=symbol_code)
    if symbol == None:
        raise ValueError(f"Symbol {symbol_code} is not known !")

    del_st = text(
        f"DELETE FROM CANDLE WHERE SK_SYMBOL={symbol.sk_symbol}")
    return con.execute(del_st)


def check_candles_last_months(session: Session, con: Connection, symbol_code: str, timeframe: int = 1440) -> pd.DataFrame:
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

    symbol = get_symbol(session=session, symbol_code=symbol_code)
    if symbol == None:
        raise ValueError(f"Symbol {symbol_code} is not known !")

    query = text(f""" SELECT  STRFTIME('%Y-%m',OPEN_DATETIME) AS MONTH ,COUNT(*) AS NB,
                MIN(OPEN_DATETIME),MAX(OPEN_DATETIME),MIN(CLOSE),MAX(CLOSE)   FROM CANDLE 
                WHERE SK_SYMBOL={symbol.sk_symbol} AND TIMEFRAME={timeframe} 
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


def insert_object(session: Session, orm_class: Type[T], obj: Any) -> T:
    """
    Insert an object into the DB using its ORM class.

    Args:
        session (Session): SQLAlchemy session.
        orm_class (Type[Base]): The ORM class corresponding to the table.
        obj (Any): The object to insert. May be:
            - an instance of orm_class (will be added as-is),
            - a dict of column names -> values,
            - an object exposing a to_dict() method,
            - or any plain object (public attributes will be used).

    Returns:
        Base: The inserted ORM instance (refreshed).

    Raises:
        SQLAlchemyError: If the insert/commit fails (the session will be rolled back).
        TypeError: If data cannot be converted to the ORM constructor args.
    """
    # prepare data / instance
    if isinstance(obj, orm_class):
        instance = obj
    else:
        if isinstance(obj, dict):
            data = obj
        elif hasattr(obj, "to_dict"):
            data = obj.to_dict()
        else:
            data = {k: v for k, v in vars(obj).items() if not k.startswith("_") and not callable(v)}
        instance = orm_class(**data)

    try:
        session.add(instance)
        session.commit()
        session.refresh(instance)
        return instance
    except SQLAlchemyError:
        session.rollback()
        raise


def insert_link(session: Session, orm_class: Type[T], key_values: Dict[str, Any]) -> T:
    """
    Idempotently insert a row into a simple join/link table.

    Args:
        session (Session): SQLAlchemy session.
        orm_class (Type[Base]): Join-table ORM class (e.g. CombiModels).
        key_values (Dict[str, Any]): Mapping of ORM attribute/column names to values used
            to check for an existing link (e.g. {'sk_strategy': 1, 'sk_model': 2}).

    Returns:
        Base: The existing or newly created ORM instance.

    Raises:
        ValueError: If key_values is empty.
        SQLAlchemyError: If the insert/commit fails (the session will be rolled back).
    """
    if not key_values:
        raise ValueError("key_values required")

    # check existing link
    existing = session.query(orm_class).filter_by(**key_values).first()
    if existing:
        return existing

    entry = orm_class(**key_values)
    try:
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry
    except SQLAlchemyError:
        session.rollback()
        raise

def insert_combi(session: Session, strategy: "Strategy") -> List[CombiModels]:
    """
    Insert links (strategy <-> model) for all models of a Strategy object.

    Args:
        session (Session): SQLAlchemy session.
        strategy (Strategy): Strategy object. Must have .id and .models (iterable of model objects).

    Returns:
        List[CombiModels]: List of existing or newly created CombiModels ORM instances.

    Raises:
        ValueError: If strategy or required ids are missing.
        SQLAlchemyError: If DB insert/commit fails (session will be rolled back).
    """
    if strategy is None:
        raise ValueError("strategy must be provided")

    sk_strategy = getattr(strategy, "id", None) #or getattr(strategy, "sk_strategy", None)
    if sk_strategy is None:
        raise ValueError("strategy.id (sk_strategy) must be set on the Strategy object before calling insert_combi")

    models = getattr(strategy, "models", None)
    if not models:
        raise ValueError("strategy.models must be a non-empty iterable of model objects")

    created: List[CombiModels] = []
    for m in models:
        sk_model = getattr(m, "id", None) #or getattr(m, "sk_model", None)
        if sk_model is None:
            raise ValueError(f"Model in strategy.models is missing id/sk_model: {m!r}")

        # insert_link is idempotent and handles existing links
        entry = insert_link(session, CombiModels, {"sk_strategy": sk_strategy, "sk_model": sk_model})
        created.append(entry)

    return created

_INT_FIELDS = {
    "final_value", "profit", "initial_cash", "max_drawdown_val",
    "nb_trades", "nb_sells", "nb_wins", "nb_losses", "nb_win_streak", "nb_loss_streak","avg_gain","avg_risk"
}

_FLOAT2_FIELDS = {
    "total_commission", "return_pct", "max_drawdown_pct",
    "sharpe_ratio", "calmar_ratio", "avg_trade_return", "win_rate","profit_factor","risk_reward_win","risk_reward_loss"
}

def _normalize_decimal(value: Decimal, key: str, int_round: bool, max_decimals: int) -> Any:
    if key in _INT_FIELDS and int_round:
        return int(value.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if key in _FLOAT2_FIELDS:
        return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
    zeros = "0" * max_decimals
    return float(value.quantize(Decimal(f"1.{zeros}"), rounding=ROUND_HALF_UP))


def _normalize_float(value: float, key: str, int_round: bool) -> Any:
    if key in _INT_FIELDS and int_round:
        return int(round(value))
    if key in _FLOAT2_FIELDS:
        return round(value, 2)
    return value


def _normalize_field(key: str, value, int_round: bool = True, max_decimals: int = 5) -> Any:
    """Normalize a single value for SQL insertion.
    - convert pandas/numpy scalars -> python
    - round floats to int for integer fields, else round floats to 2 decimals
    - serialize lists/dicts to JSON

    Args:
        key (str): The field/column name (used to determine rounding).
        int_round (bool): Whether to round integer fields to int.
        max_decimals (int): Maximum number of decimal places for float rounding.
    Returns:
        Any: The normalized value.
    """
    if value is None:
        return None

    # pandas Timestamp -> ISO string (DB columns are text)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    # numpy scalar -> python scalar
    if isinstance(value, np.generic):
        value = value.item()

    # Decimal -> format or quantize
    if isinstance(value, Decimal):
        return _normalize_decimal(value, key, int_round, max_decimals)

    # datetime -> ISO string
    if isinstance(value, dt):
        return value.strftime("%Y-%m-%d %H:%M:%S")

    # floats -> round
    if isinstance(value, float):
        return _normalize_float(value, key, int_round)

    # ints/bools/str are fine
    if isinstance(value, (int, bool, str)):
        return value

    # lists/tuples/dicts -> JSON string
    if isinstance(value, (list, tuple, dict)):
        try:
            return json.dumps(value, default=str, separators=(",", ":"))
        except Exception:
            return str(value)

    # fallback
    return str(value)



def insert_bt_result(session: Session, sk_symbol: int, sk_scenario: int,
                     date_start:str, date_end:str, unit_time:str, log_file: str, extra: Optional[Dict[str, Any]] = None):
    """
    Insert a BT_RESULT row.

    Args:
        session (Session): SQLAlchemy session.
        sk_symbol (int): campaign symbol primary key.
        sk_scenario (int): scenario primary key.
        date_start (str): start date of the backtest.
        date_end (str): end date of the backtest.
        unit_time (str): time unit of the backtest.
        log_file (str): path to the log file.
        extra (Optional[Dict]): any additional columns to include.

    Returns:
        The inserted BT_RESULT ORM instance (via insert_object) or raises on error.
    """
    data = { 
        "sk_symbol": sk_symbol,
        "sk_scenario": sk_scenario,
        "date_start": date_start,
        "date_end": date_end,
        "unit_time": unit_time,
        "log_filename": log_file, 
    }
    data["date_test"] = data.get("date_test") or dt.now()

    if extra:
        for k, v in extra.items():
            key = k.lower()  # ensure lower-case attribute keys (db_models uses snake_case)
            data[key] = _normalize_field(key, v)

    return insert_object(session, BtResult, data)


if __name__ == "__main__":
    symbol = "OVH.PA"
    model_name = "CW8_DCA_CLOSE_1D_V1_lab_perf_21d_LSTM_CLASS"
    timeframe = 1440
    db_name = "dataset_market.db"
    candle_name = "candle_CW8.db"
    con_CW8 = get_connection(r"D:\Projets\Data\sqlite\candle_CW8.db")

    con_fwk = get_connection(
        str_db_path=r"D:\Projets\Data\sqlite\dataset_market.db")
    my_session_maker = sessionmaker(bind=con_fwk)
    session = my_session_maker()

    sym = get_symbol(session=session, symbol_code=symbol)
    if sym is None:
        raise ValueError(f"Symbol {symbol} not found")
    print(f"SK du symbol {symbol} : {sym.sk_symbol} {sym.name=}")
    if con_fwk is not None:
        df_stocks = get_info_all_stock(con_fwk)
        print(df_stocks.head())

