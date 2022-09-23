import backtrader as bt
from backtrader.feeds import PandasData
from datetime import timedelta


#OHLCV = ['open', 'high', 'low', 'close', 'volume']
# class to define the columns we will provide


class SignalData(PandasData):
    """
    Define pandas DataFrame structure
    """
    cols = ['open', 'high', 'low', 'close','volume', 'Predict', 'SL', 'TP']

    # create lines
    lines = tuple(cols)

    # define parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())

# code based on Sabir Jana medium page
# https://medium.com/analytics-vidhya/ml-classification-algorithms-to-predict-market-movements-and-backtesting-2382fdaf7a32
# define backtesting strategy class


class BTMLStrategy(bt.Strategy):
    """A class to backtest strategies using Backtrader
        class based on Sabir Jana medium page
    Args:
        bt (bt.Strategy): the  backtrader strategy
    """
    params = dict()

    def __init__(self):
        # keep track of open, close prices and predicted value in the series
        self.data_predict = self.datas[0].Predict
        self.data_open = self.datas[0].open
        self.data_close = self.datas[0].close
        self.data_SL = self.datas[0].SL
        self.data_TP = self.datas[0].TP

        # keep track of pending orders/buy price/buy commission
        self.order = None
        self.price = None
        self.comm = None
        self.orefs = list()

    # logging function
    def log(self, txt):
        """Logging function"""
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt}, {txt}")

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # order already submitted/accepted - no action required
            return
        # report executed order
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"BUY EXECUTED {order.ref} --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}, Size: {order.created.size:9.4f}"
                         )
                self.price = order.executed.price
                self.comm = order.executed.comm
            else:
                self.log(f"SELL EXECUTED {order.ref} --- Price: {order.executed.price:.2f}, Cost: {order.executed.value:.2f},Commission: {order.executed.comm:.2f}, Size: {order.created.size:9.4f}"
                         )
        # report failed order
        elif order.status in [order.Canceled, order.Margin,
                              order.Rejected]:
            self.log(
                f"Order {order.getordername()} {order.getstatusname()} {order.ref}  ")
        
        if not order.alive() and order.ref in self.orefs:
            self.orefs.remove(order.ref)
        # set no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(
            f"OPERATION RESULT --- Gross: {trade.pnl:.2f}, Net: {trade.pnlcomm:.2f}")

    # We have set cheat_on_open = True.This means that we calculated the signals on day t's close price,
    # but calculated the number of shares we wanted to buy based on day t+1's open price.
    def next_open(self):
        if not self.position:
            if self.data_predict >= 1:
                size_max = int(self.broker.getcash() / self.datas[0].open)  # calculate the max number of shares ('all-in')
                size=int(2000/(self.datas[0].open-self.data_SL[0])) #risk 2%  fix <--- TODO !! 
                if size > size_max:
                    size=size_max
                if size >= 1:
                    # buy order
                    self.long_buy_order = self.buy_bracket(
                        size=size, exectype=bt.Order.Market, stopprice=self.data_SL[0], stopexec=bt.Order.Stop, limitprice=self.data_TP[0], limitexec=bt.Order.Limit,)
                    self.limit_date = self.datas[0].datetime.datetime(
                        0) + timedelta(days=7)
                    self.orefs = [ord.ref for ord in self.long_buy_order]
                    self.log(
                        f"BUY CREATED --- Size: {size}, Cash: {self.broker.getcash():.2f}, Stop: {self.data_SL[0]}, TP: {self.data_TP[0]}, date limit {self.limit_date}")
                else:
                    self.log(
                        f"ERROR SIZING cash {self.broker.getcash():.2f}, size {size}")

        else:
            # if date limit passed
            if self.limit_date <= self.datas[0].datetime.datetime(0):
                # sell order
                self.log(f"CLOSE POSITION")
                self.close()
                for ord in self.broker.orders:
                    if ord.status in [ord.Submitted, ord.Accepted]:
                        self.cancel(ord)



