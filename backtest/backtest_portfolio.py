import os
import pandas as pd
import logging
import numpy as np
from datetime import datetime

BUY_ORDER='BUY'
PATH_LOGS =os.path.dirname(os.path.abspath(__file__))+'/logs/'


class Position:
    def __init__(self, asset_code: str, entry_price: float, quantity: int, avg_cost: float, stop_loss: float, order : str=BUY_ORDER, commission: float = 0.0, comment: str = None):
        """
        Initialize a new Position.

        Args:
            asset_code: The code of the asset.
            entry_price: The entry price of the asset.
            quantity: The number of stocks.
            unit_cost: The unit cost of the asset.
            stop_loss: The stop loss of the asset.
            order: The order type (BUY or SELL).
            commission: The commission for the transaction in percent.
            comment: A comment for the position.
        """
        self.order = order
        self.asset_code = asset_code
        self.entry_price = entry_price
        self.quantity = quantity
        self.avg_cost = avg_cost
        self.stop_loss = stop_loss
        self.last_price = entry_price
        self.comment = comment
        self.initial_stop_loss = stop_loss
        self.initial_quantity = quantity
        self.commission = commission
        self.initial_risk = commission+(entry_price - stop_loss) * quantity
        self.initial_risk_ratio = self.initial_risk / (commission+(entry_price * quantity)) if entry_price * quantity >0 else 0
        

    def __str__(self):
        return f'Position: asset_code={self.asset_code}, entry_price={self.entry_price}, quantity={self.quantity}, avg_cost={self.avg_cost}, stop_loss={self.stop_loss}, last_price={self.last_price}, comment={self.comment}'

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
        self.initial_cash = initial_cash
        self.positions = {}
        self.cash = initial_cash
        self.max_positions = max_positions
        self.scale_up = scale_up
        self.nb_positions = 0
        self.nb_trades = 0
        self.nb_sells=0
        self.nb_wins = 0
        self.nb_losses = 0
        self.win_streak = 0
        self.loss_streak = 0
        self.max_win_streak = 0
        self.max_loss_streak = 0
        self.total_commission = 0
        self.cum_gain = 0.0
        self.cum_loss = 0.0
        self.cum_risk = 0.0
        self.avg_risk_reward_ratio_win = 0.0
        self.avg_risk_reward_ratio_loss = 0.0
        self.value = initial_cash
        self.trade_log = []  # List of dicts or DataFrame for trade details
        self.metrics = {}    # Dict for summary metrics
        self.file_path=None


        #history
        self.history = pd.DataFrame(columns=['date','cash','nb_positions','value'])

        # Set up logging
        self.logger = logging.getLogger('Portfolio')
        self.logger.setLevel(logging.INFO)

        if log_to_file:
            self.file_path = PATH_LOGS+'bt_portfolio_'+ datetime.now().strftime('%Y%m%d_%H%M%S') +'.log'
            handler = logging.FileHandler(self.file_path,mode='w')
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
        pretty_positions = '\n'.join([f'{asset_code}: {pos.quantity} at {round(pos.avg_cost,2)}' for asset_code, pos in self.positions.items()])
        return f'Portfolio: cash={round(self.cash,2)}, nb_positions={self.nb_positions}, value={round(self.value,2)}, positions={pretty_positions} '
    
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
    
    def close_logger(self):
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)

    def pretty_print_in_log(self,dt_action:pd.Timestamp):
        """ Print the portfolio in the log.

        Args:
            dt_action (pd.Timestamp): The date of the action.
        """
        pretty_positions = '\n'.join([f'{asset_code}: {pos.quantity} at {round(pos.avg_cost,2)}' for asset_code, pos in self.positions.items()])
        self.log(dt_action,f'Portfolio : cash={round(self.cash,2)}, nb_positions={self.nb_positions}, value={round(self.value,2)}, positions={pretty_positions}',level=logging.INFO)

    def log_trade(self, trade_dict: dict):
        """ Log a trade.
        Args:
            trade_dict (dict): The trade details to log.
        """
        self.trade_log.append(trade_dict)

    def compute_metrics(self):
        """ Compute the metrics of the portfolio.
        """
        # Example: compute final value, max drawdown, etc.
        self.metrics['final_value'] = self.value
        self.metrics['initial_cash'] = self.initial_cash
        self.metrics['profit'] = self.value - self.initial_cash
        self.metrics['return_pct'] = (self.value - self.initial_cash)*100 / self.initial_cash
        self.metrics['max_drawdown_val'] = self.history['value'].cummax().sub(self.history['value']).max() #???
        self.metrics['max_drawdown_pct'] = self.metrics['max_drawdown_val']*100 / self.history['value'].cummax().max() if self.history['value'].cummax().max()>0 else 0
        self.metrics['nb_sells'] = self.nb_sells
        self.metrics['nb_trades'] = self.nb_trades
        self.metrics['total_commission'] = self.total_commission
        self.metrics['nb_wins'] = self.nb_wins
        self.metrics['nb_losses'] = self.nb_losses
        self.metrics['max_win_streak'] = self.max_win_streak
        self.metrics['max_loss_streak'] = self.max_loss_streak
        self.metrics['win_rate'] = self.nb_wins / self.nb_sells if self.nb_sells > 0 else 0
        # self.metrics['loss_rate'] = self.nb_losses / self.nb_sells if self.nb_sells > 0 else 0
        self.metrics['sharpe_ratio'] = (self.history['value'].pct_change().mean() / self.history['value'].pct_change().std() * np.sqrt(252)) if self.history['value'].pct_change().std() != 0 else 0
        self.metrics['calmar_ratio'] = self.metrics['return_pct'] / abs(self.metrics['max_drawdown_pct']) if self.metrics['max_drawdown_pct'] != 0 else 0
        self.metrics['avg_trade_return'] = (self.value - self.initial_cash) / self.nb_sells if self.nb_sells > 0 else 0
        self.metrics['profit_factor'] = self.cum_gain / self.cum_loss if self.cum_loss != 0 else 0
        self.metrics['avg_gain'] = (self.cum_gain ) / self.nb_sells if self.nb_sells > 0 else 0
        self.metrics['avg_risk'] = (self.cum_risk ) / self.nb_sells if self.nb_sells > 0 else 0
        self.metrics['risk_reward_win'] = self.avg_risk_reward_ratio_win
        self.metrics['risk_reward_loss'] = self.avg_risk_reward_ratio_loss

        # print('Portfolio metrics OK')


    def update_value(self,dt_action:pd.Timestamp):
        """ Update the value of the portfolio.

        Args:
            dt_action (pd.Timestamp): The date of the action.
        """
        value = self.cash
        for asset_code, pos in self.positions.items():
            value += pos.quantity * pos.last_price
        self.value = value

        if self.history.empty:
            self.history = pd.DataFrame({'date':dt_action,'cash':self.cash,'nb_positions':self.nb_positions,'value':self.value},index=[0])
        else:
            self.history=pd.concat([self.history,pd.DataFrame({'date':dt_action,'cash':self.cash,'nb_positions':self.nb_positions,'value':self.value},index=[0])],ignore_index=True)


    def buy(self, asset_code: str,dt_action:pd.Timestamp, quantity: int, price: float,sl:float,
              commission: float = 0.0, message:str=''):
        """
        Buy stocks and add them to the portfolio.

        Args:
            asset_code: The code of the asset to buy.
            dt_action: The date of the action.
            quantity: The number of stocks to buy.
            price: The price of the stocks.
            sl: The stop loss of the asset.
            commission: The commission for the transaction in percent.
            message: A message to log. Default ''.
        """

        if quantity <= 0 or price <= 0:
            self.log(dt_action,f'Invalid number of stocks or price: {quantity} {price}',level=logging.WARNING)
            return

        cost = quantity * price 
        val_commission = cost * commission
        cost += val_commission
        if cost > self.cash:
            self.log(dt_action,f'Not enough cash to buy {quantity} of {asset_code}',level=logging.WARNING)
            return

        if asset_code in self.positions:
            if self.scale_up:
                tmp_avg_cost = self.positions[asset_code].avg_cost*self.positions[asset_code].quantity
                tmp_avg_cost += cost
                self.cash -= cost
                self.positions[asset_code].quantity += quantity
                self.positions[asset_code].avg_cost=tmp_avg_cost/self.positions[asset_code].quantity
                self.nb_trades += 1
                self.total_commission += val_commission
                self.log(dt_action,f'Scale up {quantity} of {asset_code}, cost: {round(cost,2)}, remaining cash: {round(self.cash,2)} {message}',level=logging.INFO)
                self.log_trade({
                    'date': dt_action,
                    'action': 'buy',
                    'asset_code': asset_code,
                    'quantity': quantity,
                    'price': price,
                    'sl': sl,
                    'cost': round(cost,2),
                    'avg_cost': self.positions[asset_code].avg_cost,
                    'commission': round(val_commission,2),
                    'message': message
                })
            else :
                self.log(dt_action,f'No scale up and {asset_code} already in positions. Not buying.',level=logging.WARNING)
        else:

            position = Position(order=BUY_ORDER,asset_code=asset_code, entry_price=price, quantity=quantity, avg_cost=cost/quantity, stop_loss=sl, commission=commission)  
            self.positions[asset_code] = position
            self.cash -= cost
            self.nb_positions += 1
            self.nb_trades += 1
            self.total_commission += val_commission
            risk = (price - sl) * quantity
            self.cum_risk += (price - sl) * quantity
            self.log(dt_action,f'Bought {quantity} of {asset_code}, cost: {round(cost,2)}, remaining cash: {round(self.cash,2)} {message}',level=logging.INFO)
            self.log_trade({
                'date': dt_action,
                'action': 'buy',
                'asset_code': asset_code,
                'quantity': quantity,
                'price': price,
                'sl': sl,
                'risk': round(risk,2),
                'cost': round(cost,2),
                'avg_cost': position.avg_cost,
                'commission': round(val_commission,2),  
                'message': message
            })

    def sell(self, asset_code: str,dt_action:pd.Timestamp, price: float, commission: float = 0.0, message:str=''):
        """
        Sell stocks and remove them from the portfolio.

        Args:
            asset_code: The code of the asset to sell.
            dt_action: The date of the action.
            price: The price of the stocks.
            commission: The commission for the transaction  in percent.
            message: A message to log. Default ''.
        """
        if asset_code in self.positions:
            pos = self.positions[asset_code]
            nb_stocks = pos.quantity
            revenue = nb_stocks * price 
            val_commission = revenue * commission
            revenue -= val_commission
            avg_revenue = revenue/nb_stocks
            self.cash += revenue
            self.total_commission += val_commission
            profit = revenue - (pos.avg_cost * nb_stocks)
            risk_reward_ratio= (profit) / pos.initial_risk if pos.initial_risk !=0 else 0
            self.nb_positions -= 1 
            self.nb_trades += 1
            self.nb_sells += 1


            if profit > 0:
                self.nb_wins += 1
                self.win_streak += 1
                self.loss_streak = 0
                self.max_win_streak = max(self.max_win_streak, self.win_streak)
                self.cum_gain += profit
                self.avg_risk_reward_ratio_win = ((self.avg_risk_reward_ratio_win * (self.nb_wins-1)) + risk_reward_ratio) / (self.nb_wins)
            else:
                self.nb_losses += 1
                self.loss_streak += 1
                self.win_streak = 0
                self.max_loss_streak = max(self.max_loss_streak, self.loss_streak)
                self.cum_loss += -profit
                self.avg_risk_reward_ratio_loss = ((self.avg_risk_reward_ratio_loss * (self.nb_losses-1)) + risk_reward_ratio) / (self.nb_losses)

            # self.log(dt_action,f'{profit=} {risk_reward_ratio=} {self.avg_risk_reward_ratio_win=} {self.avg_risk_reward_ratio_loss=}',level=logging.INFO)
            self.log(dt_action,f'Sold {nb_stocks} of {asset_code}, revenue: {round(revenue,2)}, remaining cash: {round(self.cash,2)}, RR {risk_reward_ratio} {message}',level=logging.INFO)
            self.log_trade({
                'date': dt_action,
                'action': 'sell',
                'asset_code': asset_code,
                'quantity': nb_stocks,
                'price': price,
                'revenue': round(revenue,2),
                'avg_revenue': round(avg_revenue,2),
                'commission': round(val_commission, 2),
                'message': message
            })

            del self.positions[asset_code]

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
            portfolio.sell(asset_code=asset_code,dt_action=dt, price=df_in.loc[(dt, asset_code)]['low'], commission=commission, message=f"SL {df_in.loc[(dt, asset_code)]['comment']}")
        else: # if a position is in the portfolio, sell it if the exit signal is True
            if df_in.loc[(dt, asset_code)]['exit']:
                portfolio.sell(asset_code=asset_code,dt_action=dt, price=df_in.loc[(dt, asset_code)]['price'], commission=commission, message=f"{df_in.loc[(dt, asset_code)]['comment']}")
            else: #update last_price
                if df_in.loc[(dt, asset_code)]['price']>0:
                    pos.last_price = df_in.loc[(dt, asset_code)]['price'] 
                    

    return portfolio

def backtest_entry_strategy(df_in:pd.DataFrame, portfolio:Portfolio, dt:pd.Timestamp, commission:float=0.0,
                             fixe_quantity:bool=True) ->Portfolio:
    """
    Backtests an entry strategy using the given DataFrame and portfolio.

    Args:
        df_in (pandas.DataFrame): The input DataFrame containing the trading signals.
        portfolio (Portfolio): The portfolio object containing the remaining positions and cash.
        dt (pd.Timestamp): The date of the action.
        commission (float, optional): The commission fee for each trade. Defaults to 0.0.
        fixe_quantity (bool, optional): Whether to use a fixe quantity of stocks in df_in else calculated. Defaults to True.

    Returns:
        Portfolio: The portfolio object containing the remaining positions and cash.
    """
    # iterate over the rows of the DataFrame and buy the stock if the entry signal is True
    for index, row in df_in[df_in['entry']].sort_values('priority').iterrows():
        if portfolio.nb_positions >= portfolio.max_positions:
            break
        else:
            if fixe_quantity:
                quantity=row['quantity']
            else:
                quantity=np.floor(portfolio.cash/(portfolio.max_positions-portfolio.nb_positions)/row['price'])
            portfolio.buy(asset_code=row.name[1],dt_action=dt, quantity=quantity, price=row['price'], sl=row['sl'], commission=commission, message=f"{row['comment']}")

    return portfolio

def backtest_strategy_portfolio(df_in:pd.DataFrame, initial_cash:float=10000.0, commission:float=0.0,options: dict = None, log_to_file:bool=False,freq_print:int=0) ->Portfolio:
    """
    Backtests a trading strategy using the given DataFrame and initial cash.
    df_in must have a MultiIndex with the date and asset_code as the index.
    df_in must have the following columns: entry, exit, price, low, sl, priority(, quantity).

    Args:
        df_in (pandas.DataFrame): The input DataFrame containing the trading signals.
        initial_cash (float): The initial cash available for trading. Defaults to 10000.0.
        commission (float, optional): The commission fee for each trade in percent. Defaults to 0.0.
        options (dict, optional): A dictionary containing the options for the backtest. Options are :
            - max_positions (int, optional): The maximum number of positions that can be held in the portfolio. Defaults to 10.
            - scale_up (bool, optional): Whether to scale up the position if the asset is already in the portfolio. Defaults to False.
            - fixe_quantity (bool, optional): Whether to use a fixe quantity of stocks in df_in else calculated. Defaults to True.
            - sell_all (bool, optional): Whether to sell all remaining positions at the end. Defaults to False.
        log_to_file (bool, optional): Whether to log to a file. If False, logs to the console. Defaults to False.
        freq_print (int, optional): The frequency at which to print the portfolio. If 0, does not print. Defaults to 0.

    Returns:
        Portfolio: The portfolio object containing the remaining positions and cash.
    """

    max_positions=options.get('max_positions',10)
    scale_up=options.get('scale_up',False)
    fixe_quantity=options.get('fixe_quantity',True)
    sell_all=options.get('sell_all',False)

    df_sorted = df_in.sort_index()

    # check if the DataFrame is empty
    if df_sorted.empty:
        raise ValueError("The input DataFrame is empty.")
    
    # Check if column comment exists and add a default column with None if not
    if 'comment' not in df_sorted.columns:
        df_sorted['comment'] = None

    portfolio = Portfolio(initial_cash=initial_cash, max_positions=max_positions, scale_up=scale_up, log_to_file=log_to_file)
    cnt_print=0

    # iterate over the date index[0] of the DataFrame and get a dataframe of the rows for this date   
    for dt, df_group in df_sorted.groupby(level=0):
        
        portfolio = backtest_exit_strategy(df_group, portfolio, dt, commission)

        # if max positions is not reached
        if portfolio.nb_positions < max_positions:
            portfolio = backtest_entry_strategy(df_group, portfolio, dt, commission,fixe_quantity)

        portfolio.update_value(dt)

        cnt_print+=1
        if freq_print>0 and cnt_print==freq_print:
            portfolio.pretty_print_in_log(dt)
            cnt_print=0

    # sell all remaining positions
    if sell_all:
        for asset_code in portfolio.positions.copy():
            portfolio.sell(asset_code=asset_code,dt_action=dt, price=df_group.loc[(dt, asset_code)]['price'], commission=commission, message=' LIQUIDATE')

    portfolio.update_value(dt)
    portfolio.compute_metrics()
    portfolio.close_logger()

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
        'comment': [None, None, '(comm 1)', None, None, '(comm 2)', '(comm 3)', '(comm 4)', None, None, None, None]
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
    df['quantity']=np.floor(1000/df['price'])

    # Test the backtest_strategy function
    initial_cash = 2000
    commission = 0.005
    options = {'max_positions': 2, 'scale_up': False, 'sell_all': False, 'fixe_quantity':False}
    remaining_portfolio = backtest_strategy_portfolio(df_in=df, initial_cash=initial_cash, commission=commission,options=options,freq_print=4,log_to_file=False)
    print(f'Remaining portfolio: {remaining_portfolio}')
    print(f'{remaining_portfolio.nb_trades= } {remaining_portfolio.total_commission= }')
    print(remaining_portfolio.metrics)
    print(remaining_portfolio.history)