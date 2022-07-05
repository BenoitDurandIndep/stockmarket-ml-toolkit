from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from decouple import AutoConfig
from pandas import DataFrame as df
import pandas

config = AutoConfig(search_path=".env")
ALPHAVANTAGE_KEY = config("ALPHAVANTAGE_KEY", cast=str)


class GetDataAlphaVantage:
    """
    A class to get data from Alpha Vantage https://www.alphavantage.co/documentation 
    in panda dataframes 
    ---
    Attributes
    -----------
    ts : the TimeSeries from alpha_vantage

    Methods : 
    ---------
    get_stock(self, symbol: str, interval: str, adjusted: bool = True, compact: bool = True) -> tuple[df, str]:
        Returns the dataframe for the symbol and the interval

    """

    def __init__(self) -> None:
        """
        Constructs the timeseries
        """
        self._ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format="pandas")

    def get_stock(self, symbol: str, interval: str, adjusted: bool = True, compact: bool = True) -> tuple[df, str]:
        """
        Returns the dataframe for the symbol and the interval

            Parameters : 
                symbol(str) : a symbol or a list of symbol
                interval(str) : an interval (1min, 5min, 15min, 30min, 1h, 1d, 1w, 1mon)
                adjusted(bool) : if price is adjusted, default True
                compact(bool) : if True only 100 data, default  True

            Returns:
                a tuple with 
                    dataframe : the dataframe with the data
                    string : the metadata of the request
        """

        match interval:
            case '1m':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval=interval, adjusted=adjusted)
            case '5m':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval=interval, adjusted=adjusted)
            case '15m':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval=interval, adjusted=adjusted)
            case '30m':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval=interval, adjusted=adjusted)
            case '1h':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval="60m", adjusted=adjusted)
            case '1d':
                df_result, metadata = self._ts.get_daily(
                    symbol=symbol, outputsize=compact)
            case '1w':
                if(adjusted):
                    df_result, metadata = self._ts.get_weekly_adjusted(
                        symbol=symbol)
                else:
                    df_result, metadata = self._ts.get_weekly(symbol=symbol)
            case '1mon':
                if(adjusted):
                    df_result, metadata = self._ts.get_monthly_adjusted(
                        symbol=symbol)
                else:
                    df_result, metadata = self._ts.get_monthly(symbol=symbol)
            case _:
                df_result = pandas.DataFrame()
                metadata = "INTERVAL NOT FOUND!"

        return df_result, metadata


if __name__ == "__main__":
    get_data_alpha = GetDataAlphaVantage()

    data_tte_1w, meta = get_data_alpha.get_stock(
        symbol="TTE.PAR", interval="1w", adjusted=True)
    pprint(data_tte_1w.head())
    pprint(meta)
