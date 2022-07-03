#import logging
#from datetime import date
#from pathlib import Path
from nis import match
from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from decouple import AutoConfig
from pandas import DataFrame as df
import pandas

config = AutoConfig(search_path=".env")
ALPHAVANTAGE_KEY = config("ALPHAVANTAGE_KEY", cast=str)
#LOG_DIR=config("LOG_DIR", cast=str)
SYMBOL = "TTE.PAR"

ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format="pandas")
data, metadata = ts.get_weekly_adjusted(symbol=SYMBOL)

pprint(data.head())
pprint(metadata)

# logging.basicConfig(level=logging.INFO,
# filename=Path(LOG_DIR) / f"get_data_alphavantage_{date.today().strftime('%Y%m%d')}.log"
# )

# LOGGER=logging.getLogger()

class GetDataAlphaVantage :
    """
    A class to get data from Alpha Vantage https://www.alphavantage.co/documentation 
    in panda dataframes 
    ---
    Attributes
    -----------

    Methods
    ---------

    """
    def __init__(self) -> None:
        ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format="pandas")
    
    def get_stock(self,symbol:str,interval:str,adjusted:bool=True,compact:bool=True)-> df:
        """
        Returns the dataframe for the symbol and the interval

            Parameters : 
                symbol(str) : a symbol or a list of symbol
                interval(str) : an interval (1min, 5min, 15min, 30min, 1h, 1d, 1w, 1m)
                adjusted(bool) : if price is adjusted, default True
                compact(bool) : if True only 100 data, default  True

            Returns:
                dataframe : the dataframe with the data
        """
        df_result=pandas.DataFrame()

        match interval:
            case '1m':
                df_result=ts.get_intraday_extended(symbol=SYMBOL, interval=interval,adjusted=adjusted)


        return df_result
