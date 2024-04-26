import os
import pandas as pd
import logging
import numpy as np
from datetime import datetime

# logging.basicConfig(filename='backtest.log',
#                     filemode='a',
#                     format='%(asctime)s %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     level=logging.INFO)

BUY_ORDER='BUY'
PATH_LOGS =os.path.dirname(os.path.abspath(__file__))+'/logs/'


class Position:
    def __init__(self, asset_code: str, entry_price: float, quantity: int, avg_cost: float, stop_loss: float, order : str=BUY_ORDER):
        """
        Initialize a new Position.

        Args:
            asset_code: The code of the asset.
            entry_price: The entry price of the asset.
            quantity: The number of stocks.
            unit_cost: The unit cost of the asset.
            stop_loss: The stop loss of the asset.
            order: The order type (BUY or SELL).
        """
        self.order = order
        self.asset_code = asset_code
        self.entry_price = entry_price
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.stop_loss = stop_loss

    def __str__(self):
        return f'Position: asset_code={self.asset_code}, entry_price={self.entry_price}, quantity={self.quantity}, avg_cost={self.avg_cost}, stop_loss={self.stop_loss}'

class Portfolio:
    def __init__(self, initial_cash : float =10000, max_positions : int =10, scale_up : bool=False, log_to_file: bool = False):
        """
        Initialize a new Portfolio.

        Args:
            initial_cash: The initial cash in the portfolio.
            max_positions: The maximum number of positions that can be held in the portfolio.
            scale_up: Whether to scale up the position if the asset is already in the portfolio.
            log_to_file: Whether to log to a file. If False, logs to the console.
        """
        self.positions = {}
        self.cash = initial_cash
        self.max_positions = max_positions
        self.scale_up = scale_up
        self.nb_positions = 0

        # Set up logging
        self.logger = logging.getLogger('Portfolio')
        self.logger.setLevel(logging.INFO)

        if log_to_file:
            handler = logging.FileHandler(PATH_LOGS+'bt_portfolio_'+ datetime.now().strftime('%Y%m%d') +'.log',mode='w')
        else:
            handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.INFO)
        self.logger.addHandler(handler)

        if log_to_file:
            print(f'Logging to file: {PATH_LOGS}')
            self.logger.info(f'Initial cash: {initial_cash}, max_positions: {max_positions}, scale_up: {scale_up}')

    
    def __str__(self):
        pretty_positions = '\n'.join([f'{asset_code}: {pos.quantity} at {pos.avg_cost}' for asset_code, pos in self.positions.items()])
        return f'Portfolio: cash={self.cash}, nb_positions={self.nb_positions}, positions={pretty_positions} '
    
    def __repr__(self):
        return self.__str__()
    
    def __len__(self):
        return self.nb_positions
    
    def log(self,dt_action:pd.Timestamp, message:str, level:int=logging.INFO):
        """ Log a message.

        Args:
            dt_action (pd.Timestamp): The date of the action.
            message (str): The message to log.
            level (int, optional): The logging level. Defaults to logging.INFO.
        """
        self.logger.log(level,f'{dt_action} {message}')
    
    def pretty_print_in_log(self,dt_action:pd.Timestamp):
        """ Print the portfolio in the log.

        Args:
            dt_action (pd.Timestamp): The date of the action.
        """
        pretty_positions = '\n'.join([f'{asset_code}: {pos.quantity} at {pos.avg_cost}' for asset_code, pos in self.positions.items()])
        self.log(dt_action,f'Portfolio : cash={self.cash}, nb_positions={self.nb_positions}, positions={pretty_positions}',level=logging.INFO)


    def buy(self, asset_code: str,dt_action:pd.Timestamp, nb_stocks: int, price: float,sl:float,  commission: float = 0.0):
        """
        Buy stocks and add them to the portfolio.

        Args:
            asset_code: The code of the asset to buy.
            dt_action: The date of the action.
            nb_stocks: The number of stocks to buy.
            price: The price of the stocks.
            sl: The stop loss of the asset.
            commission: The commission for the transaction.
        """

        if nb_stocks <= 0 or price <= 0:
            self.log(dt_action,f'Invalid number of stocks or price: {nb_stocks} {price}',level=logging.WARNING)
            return

        cost = nb_stocks * price + commission
        if cost > self.cash:
            self.log(dt_action,f'Not enough cash to buy {nb_stocks} of {asset_code}',level=logging.WARNING)
            return

        if asset_code in self.positions:
            if self.scale_up:
                tmp_avg_cost = self.positions[asset_code].avg_cost*self.positions[asset_code].quantity
                tmp_avg_cost += cost
                self.cash -= cost
                self.positions[asset_code].quantity += nb_stocks
                self.positions[asset_code].avg_cost=tmp_avg_cost/self.positions[asset_code].quantity
                self.log(dt_action,f'Scale up {nb_stocks} of {asset_code}, cost: {cost}, remaining cash: {self.cash}',level=logging.INFO)
            else :
                self.log(dt_action,f'No scale up and {asset_code} already in positions. Not buying.',level=logging.WARNING)
        else:

            position = Position(order=BUY_ORDER,asset_code=asset_code, entry_price=price, quantity=nb_stocks, avg_cost=price+(commission/nb_stocks), stop_loss=sl)  
            self.positions[asset_code] = position
            self.cash -= cost
            self.nb_positions += 1
            self.log(dt_action,f'Bought {nb_stocks} of {asset_code}, cost: {cost}, remaining cash: {self.cash}',level=logging.INFO)



    def sell(self, asset_code: str,dt_action:pd.Timestamp, price: float, commission: float = 0.0, message:str=''):
        """
        Sell stocks and remove them from the portfolio.

        Args:
            asset_code: The code of the asset to sell.
            dt_action: The date of the action.
            price: The price of the stocks.
            commission: The commission for the transaction.
            message: A message to log. Default ''.
        """
        if asset_code in self.positions:
            pos = self.positions[asset_code]
            nb_stocks = pos.quantity
            revenue = nb_stocks * price - commission
            self.cash += revenue
            del self.positions[asset_code]
            self.nb_positions -= 1 
            self.log(dt_action,f'Sold {message} {nb_stocks} of {asset_code}, revenue: {revenue}, remaining cash: {self.cash}',level=logging.INFO)


    # def sell_stop_loss(self,dt_action:pd.Timestamp, price: float, commission: float = 0.0):
    #     """
    #     Sell stocks that have reached the stop loss.

    #     Args:
    #         dt_action: The date of the action.
    #         price: The price of the stocks.
    #         commission: The commission for the transaction.
    #     """
    #     for asset_code, pos in self.positions.items():
    #         if price <= pos.stop_loss:
    #             self.sell(asset_code,dt_action, price, commission)

    def has_stock(self, asset_code:str) -> bool:
        """
        Check if the portfolio has a stock.

        Args :
            asset_code (str): The code of the asset.

        Returns:
            bool: True if the portfolio has the stock, False otherwise.
        """
        return asset_code in self.positions


def backtest_exit_strategy(df_in:pd.DataFrame, portfolio:Portfolio, dt:pd.Timestamp, commission:float=0.0) ->Portfolio:
    """
    Backtests an exit strategy using the given DataFrame and portfolio.

    Args:
        df_in (pandas.DataFrame): The input DataFrame containing the trading signals.
        portfolio (Portfolio): The portfolio object containing the remaining positions and cash.
        dt (pd.Timestamp): The date of the action.
        commission (float, optional): The commission fee for each trade. Defaults to 0.0.

    Returns:
        Portfolio: The portfolio object containing the remaining positions and cash.
    """
    # iterate over the rows of the DataFrame and sell the stock if the exit signal is True
    for asset_code, pos in portfolio.positions.copy().items():
        #check if the asset is in the dataframe
        if (dt, asset_code) not in df_in.index:
            portfolio.log(dt,f'Asset {asset_code} not in the dataframe',level=logging.ERROR)
            continue
        # if the stop loss of a position is below the low of the asset, sell the stock
        if pos.stop_loss >= df_in.loc[(dt, asset_code)]['low']:
            portfolio.sell(asset_code=asset_code,dt_action=dt, price=df_in.loc[(dt, asset_code)]['low'], commission=commission, message='SL')
        else: # if a position is in the portfolio, sell it if the exit signal is True
            if df_in.loc[(dt, asset_code)]['exit']:
                portfolio.sell(asset_code=asset_code,dt_action=dt, price=df_in.loc[(dt, asset_code)]['price'], commission=commission)

    return portfolio

def backtest_entry_strategy(df_in:pd.DataFrame, portfolio:Portfolio, dt:pd.Timestamp, commission:float=0.0) ->Portfolio:
    """
    Backtests an entry strategy using the given DataFrame and portfolio.

    Args:
        df_in (pandas.DataFrame): The input DataFrame containing the trading signals.
        portfolio (Portfolio): The portfolio object containing the remaining positions and cash.
        dt (pd.Timestamp): The date of the action.
        commission (float, optional): The commission fee for each trade. Defaults to 0.0.

    Returns:
        Portfolio: The portfolio object containing the remaining positions and cash.
    """
    # iterate over the rows of the DataFrame and buy the stock if the entry signal is True
    for index, row in df_in[df_in['entry']].sort_values('priority').iterrows():
        if portfolio.nb_positions >= portfolio.max_positions:
            break
        else:
            portfolio.buy(asset_code=row.name[1],dt_action=dt, nb_stocks=row['nb_stocks'], price=row['price'], sl=row['sl'], commission=commission)

    return portfolio

def backtest_strategy_portfolio(df_in:pd.DataFrame, initial_cash:float=10000.0, commission:float=0.0,max_positions:int=10, scale_up:bool=False,sell_all:bool=False, log_to_file:bool=False,freq_print:int=0) ->Portfolio:
    """
    Backtests a trading strategy using the given DataFrame and initial cash.
    df_in must have a MultiIndex with the date and asset_code as the index.
    df_in must have the following columns: entry, exit, price, low, sl, priority, nb_stocks.

    Args:
        df_in (pandas.DataFrame): The input DataFrame containing the trading signals.
        initial_cash (float): The initial cash available for trading. Defaults to 10000.0.
        commission (float, optional): The commission fee for each trade. Defaults to 0.0.
        max_positions (int, optional): The maximum number of positions that can be held in the portfolio. Defaults to 10.
        scale_up (bool, optional): Whether to scale up the position if the asset is already in the portfolio. Defaults to False.
        sell_all (bool, optional): Whether to sell all remaining positions at the end. Defaults to False.
        log_to_file (bool, optional): Whether to log to a file. If False, logs to the console. Defaults to False.
        freq_print (int, optional): The frequency at which to print the portfolio. If 0, does not print. Defaults to 0.

    Returns:
        Portfolio: The portfolio object containing the remaining positions and cash.
    """

    df_sorted = df_in.sort_index()

    portfolio = Portfolio(initial_cash=initial_cash, max_positions=max_positions, scale_up=scale_up, log_to_file=log_to_file)
    cnt_print=0

    # iterate over the date index[0] of the DataFrame and get a dataframe of the rows for this date   
    for dt, df_group in df_sorted.groupby(level=0):
        
        portfolio = backtest_exit_strategy(df_group, portfolio, dt, commission)

        # if max positions is not reached
        if portfolio.nb_positions < max_positions:
            portfolio = backtest_entry_strategy(df_group, portfolio, dt, commission)

        cnt_print+=1
        if freq_print>0 and cnt_print==freq_print:
            portfolio.pretty_print_in_log(dt)
            cnt_print=0

    # sell all remaining positions
    if sell_all:
        for asset_code in portfolio.positions.copy():
            portfolio.sell(asset_code=asset_code,dt_action=dt, price=df_group.loc[(dt, asset_code)]['price'], commission=commission, message=' LIQUIDATE')

    return portfolio



if __name__ == "__main__":
    # Create a sample DataFrame for testing
    #scenario : BUY APPLE, BUY MSFT, BUY APPLE FAIL, SELL APPLE, BUY MSFT FAIL

    data = {
        'entry': [True, False, True, False, True, False,       False, False, True, False, True, True],
        'exit': [False, False, False, True, False, False,       False, False, False, False, False, False],
        'price': [100, 120, 110, 130, 150, 140,        50, 60, 75, 80, 85, 90],
        'low': [95, 115, 105, 115, 145, 135,           45, 55, 70, 75, 90, 85],
        'sl': [90, 110, 100, 110, 140, 130,            40, 50, 65, 70, 85, 80],
        'priority': [1, 1, 1, 1, 1, 1,                  2, 2, 2, 2, 2, 2],
    }
    
    index = pd.MultiIndex.from_tuples([(pd.Timestamp('2024-01-02'), 'AAPL'), # BUY 
                                       (pd.Timestamp('2024-01-03'), 'AAPL'), 
                                       (pd.Timestamp('2024-01-04'), 'AAPL'), # SCALE UP 
                                       (pd.Timestamp('2024-01-05'), 'AAPL'), # SELL
                                       (pd.Timestamp('2024-01-08'), 'AAPL'), # BUY
                                       (pd.Timestamp('2024-01-09'), 'AAPL'), # SELL SL
                                       (pd.Timestamp('2024-01-02'), 'MSFT'),
                                       (pd.Timestamp('2024-01-03'), 'MSFT'),
                                       (pd.Timestamp('2024-01-04'), 'MSFT'), # BUY
                                       (pd.Timestamp('2024-01-05'), 'MSFT'),
                                       (pd.Timestamp('2024-01-08'), 'MSFT'),# SCALE UP 
                                       (pd.Timestamp('2024-01-09'), 'MSFT')],# SCALE UP 
                                      names=['date', 'asset_code'])
    df = pd.DataFrame(data, index=index)
    df['nb_stocks']=np.floor(1000/df['price'])

    # Test the backtest_strategy function
    initial_cash = 5000
    commission = 5
    remaining_portfolio = backtest_strategy_portfolio(df_in=df, initial_cash=initial_cash, commission=commission,scale_up=True,sell_all=False,freq_print=4,log_to_file=False)
    print(f'Remaining portfolio: {remaining_portfolio}')