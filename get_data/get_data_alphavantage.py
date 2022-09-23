import os
from pprint import pprint
from alpha_vantage.timeseries import TimeSeries
from decouple import AutoConfig
from pandas import DataFrame as df




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

    _REGION_LIST = ["United States", "United Kingdon", "Paris", "Frankfurt"]
    _TYPE_LIST = ["Equity", "ETF","FULL"]
    _CURR_LIST = ["EUR", "USD"]
    _INTERVAL_LIST=["1min", "5min", "15min", "30min", "1h", "1d", "1w", "1mon"]

    def __init__(self) -> None:
        """
        Constructs the timeseries
        """
        dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        config = AutoConfig(search_path=os.path.join(dir_path,".env"))
        ALPHAVANTAGE_KEY = config("ALPHAVANTAGE_KEY", cast=str)
        self._ts = TimeSeries(key=ALPHAVANTAGE_KEY, output_format="pandas")

    def search_symbol(self, keyword: str, type: str = "Equity", region: str = "Paris") -> tuple[df, str]:
        """
        Returns the dataframe with the list of symbol for the keyword and for the filters specified

            Parameters : 
                keyword(str) : keyword to search for
                type(str) : type of asset (Equity,ETF,FULL)
                region(bool) : marketplace (United States, United Kingdon, Paris, Frankfurt)

            Returns:
                a tuple with 
                    dataframe : the dataframe with the data fitlered
                    string : the metadata of the request
        """
        if type not in self._TYPE_LIST:
            raise ValueError(f"Type {type} is not known : {self._TYPE_LIST}")
        elif region not in self._REGION_LIST:
            raise ValueError(
                f"Region {region} is not known : {self._REGION_LIST}")

        df_result, metadata = self._ts.get_symbol_search(keywords=keyword)
        df_result = df(df_result)

        if type!="FULL":
            df_filtered = df_result[(df_result["3. type"] == type) & (
                df_result["4. region"] == region)]
        else :
            df_filtered = df_result

        return df_filtered, metadata

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
        if interval not in self._INTERVAL_LIST:
            raise ValueError(f"Interval {interval} is not known : {self._INTERVAL_LIST}")

        match interval:
            case '1m' | '5m' | '15m' | '30m':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval=interval)
            case '1h':
                df_result, metadata = self._ts.get_intraday_extended(
                    symbol=symbol, interval="60m")
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
                raise ValueError(f"Interval {interval} is not known : {self._INTERVAL_LIST}")

        return df_result, metadata


if __name__ == "__main__":
    get_data_alpha = GetDataAlphaVantage()
    symbol="MRK.PAR"
    # try:
    #     data_tte_1w, meta = get_data_alpha.get_stock(
    #         symbol=symbol, interval="1w", adjusted=True)
    #     pprint(data_tte_1w.head())
    # except ValueError:
    #     print(f"ERROR  requesting for {symbol} ")
    # pprint(meta)
    df_symb, meta = get_data_alpha.search_symbol(keyword="dsy")
    pprint(df_symb)
