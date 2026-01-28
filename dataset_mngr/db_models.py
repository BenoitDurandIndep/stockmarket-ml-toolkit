from datetime import datetime
from sqlalchemy import DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from typing import Optional


class Base(DeclarativeBase):
    pass


class Symbol(Base):
    __tablename__ = "SYMBOL"

    sk_symbol: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True)
    code : Mapped[str]
    name : Mapped[str]
    type : Mapped[str]
    region : Mapped[str]
    currency : Mapped[str]
    comment : Mapped[Optional[str]]
    code_alpha : Mapped[Optional[str]]
    code_yahoo : Mapped[Optional[str]]
    code_isin : Mapped[Optional[str]]
    active : Mapped[Optional[str]]
    tradable : Mapped[Optional[int]]


class SymbolInfo(Base):
    __tablename__ = "SYMBOL_INFO"

    sk_symbol_info: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True)
    sk_symbol: Mapped[int] = mapped_column(ForeignKey("Symbol.sk_symbol"))
    info : Mapped[str]
    update_date : Mapped[datetime]
    active_row : Mapped[Optional[int]]
    info_clean : Mapped[Optional[str]]
    sharesoutstanding : Mapped[Optional[int]]


class Campaign(Base):
    __tablename__ = "CAMPAIGN"
    sk_campaign: Mapped[int] = mapped_column("SK_CAMPAIGN", Integer, primary_key=True, autoincrement=True)
    code: Mapped[Optional[str]] = mapped_column("CODE", String(50))
    description: Mapped[Optional[str]] = mapped_column("DESCRIPTION", String(255))
    filename: Mapped[Optional[str]] = mapped_column("FILENAME", String(100))
    active: Mapped[int] = mapped_column("ACTIVE", Integer, default=1, server_default=text("1"), nullable=False)
    settings: Mapped[Optional[str]] = mapped_column("SETTINGS", String(255))
    initial_cash: Mapped[Optional[int]] = mapped_column("INITIAL_CASH", Integer)
    commission: Mapped[Optional[float]] = mapped_column("COMMISSION", Numeric)
    date_start: Mapped[Optional[str]] = mapped_column("DATE_START", String(25))
    date_end: Mapped[Optional[str]] = mapped_column("DATE_END", String(25))


class Model(Base):
    __tablename__ = "MODEL"
    sk_model: Mapped[int] = mapped_column("SK_MODEL", Integer, primary_key=True, autoincrement=True)
    sk_dataset: Mapped[int] = mapped_column("SK_DATASET", Integer)
    sk_symbol: Mapped[int] = mapped_column("SK_SYMBOL", Integer)
    sk_label: Mapped[int] = mapped_column("SK_LABEL", Integer)
    algo: Mapped[Optional[str]] = mapped_column("ALGO", String)
    name: Mapped[Optional[str]] = mapped_column("NAME", String)
    comment: Mapped[Optional[str]] = mapped_column("COMMENT", String)
    header_dts: Mapped[Optional[str]] = mapped_column("HEADER_DTS", String)
    file_name: Mapped[Optional[str]] = mapped_column("FILE_NAME", String)
    type_model: Mapped[Optional[str]] = mapped_column("TYPE_MODEL", String)
    lib_label: Mapped[Optional[str]] = mapped_column("LIB_LABEL", String)
    lib_predict_label: Mapped[Optional[str]] = mapped_column("LIB_PREDICT_LABEL", String)
    lib_proba_label: Mapped[Optional[str]] = mapped_column("LIB_PROBA_LABEL", String)


class StrategyType(Base):
    __tablename__ = "STRATEGY_TYPE"
    sk_strategy_type: Mapped[int] = mapped_column("SK_STRATEGY_TYPE", Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column("NAME", String)
    description: Mapped[Optional[str]] = mapped_column("DESCRIPTION", String)
    code_entry: Mapped[Optional[str]] = mapped_column("CODE_ENTRY", String)
    code_exit: Mapped[Optional[str]] = mapped_column("CODE_EXIT", String)
    param_entry: Mapped[Optional[str]] = mapped_column("PARAM_ENTRY", String)
    param_exit: Mapped[Optional[str]] = mapped_column("PARAM_EXIT", String)


class Strategy(Base):
    __tablename__ = "STRATEGY"
    sk_strategy: Mapped[int] = mapped_column("SK_STRATEGY", Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column("NAME", String(100))
    strat_type: Mapped[Optional[int]] = mapped_column("SK_STRATEGY_TYPE", Integer, ForeignKey("STRATEGY_TYPE.SK_STRATEGY_TYPE"))
    model_type: Mapped[Optional[str]] = mapped_column("MODEL_TYPE", String(100))
    description: Mapped[Optional[str]] = mapped_column("DESCRIPTION", String(255))
    settings: Mapped[Optional[str]] = mapped_column("SETTINGS", String, default="255")


class Scenario(Base):
    __tablename__ = "SCENARIO"
    sk_scenario: Mapped[int] = mapped_column("SK_SCENARIO", Integer, primary_key=True, autoincrement=True)
    sk_campaign: Mapped[Optional[int]] = mapped_column("SK_CAMPAIGN", Integer, ForeignKey("CAMPAIGN.SK_CAMPAIGN"))
    sk_strategy: Mapped[Optional[int]] = mapped_column("SK_STRATEGY", Integer, ForeignKey("STRATEGY.SK_STRATEGY"))
    settings: Mapped[Optional[str]] = mapped_column("SETTINGS", String(255))
    comment: Mapped[Optional[str]] = mapped_column("COMMENT", String(255))
    code: Mapped[Optional[str]] = mapped_column("CODE", String(25))


class CombiModels(Base):
    __tablename__ = "COMBI_MODELS"
    sk_combi_models: Mapped[int] = mapped_column("SK_COMBI_MODELS", Integer, primary_key=True, autoincrement=True)
    sk_model: Mapped[int] = mapped_column("SK_MODEL", Integer, ForeignKey("MODEL.SK_MODEL"))
    sk_strategy: Mapped[int] = mapped_column("SK_STRATEGY", Integer, ForeignKey("STRATEGY.SK_STRATEGY"))
    __table_args__ = (
        UniqueConstraint("SK_STRATEGY", "SK_MODEL", name="uq_strategy_model"),
    )

class BtResult(Base):
    __tablename__ = "BT_RESULT"
    sk_bt_result: Mapped[int] = mapped_column("SK_BT_RESULT", Integer, primary_key=True, autoincrement=True)
    sk_scenario: Mapped[Optional[int]] = mapped_column("SK_SCENARIO", Integer)
    sk_symbol: Mapped[Optional[int]] = mapped_column("SK_SYMBOL", Integer)
    date_start: Mapped[Optional[str]] = mapped_column("DATE_START", String)
    date_end: Mapped[Optional[str]] = mapped_column("DATE_END", String)
    unit_time: Mapped[Optional[str]] = mapped_column("UNIT_TIME", String)
    initial_cash: Mapped[Optional[int]] = mapped_column("INITIAL_CASH", Integer)
    final_value: Mapped[Optional[int]] = mapped_column("FINAL_VALUE", Integer)
    profit: Mapped[Optional[int]] = mapped_column("PROFIT", Integer)
    return_pct: Mapped[Optional[float]] = mapped_column("RETURN_PCT", Numeric)
    nb_trades: Mapped[Optional[int]] = mapped_column("NB_TRADES", Integer)
    max_drawdown_pct: Mapped[Optional[float]] = mapped_column("MAX_DRAWDOWN_PCT", Numeric)
    max_drawdown_val: Mapped[Optional[int]] = mapped_column("MAX_DRAWDOWN_VAL", Integer)
    date_test: Mapped[Optional[str]] = mapped_column("DATE_TEST", String) 
    nb_sells: Mapped[Optional[int]] = mapped_column("NB_SELLS", Integer)
    nb_trades: Mapped[Optional[int]] = mapped_column("NB_TRADES", Integer)
    nb_wins: Mapped[Optional[int]] = mapped_column("NB_WINS", Integer)
    nb_losses: Mapped[Optional[int]] = mapped_column("NB_LOSSES", Integer)
    max_win_streak: Mapped[Optional[int]] = mapped_column("NB_WIN_STREAK", Integer)
    max_loss_streak: Mapped[Optional[int]] = mapped_column("NB_LOSS_STREAK", Integer)
    total_commission: Mapped[Optional[float]] = mapped_column("TOTAL_COMMISSION", Numeric)
    sharpe_ratio: Mapped[Optional[float]] = mapped_column("SHARPE_RATIO", Numeric)
    calmar_ratio: Mapped[Optional[float]] = mapped_column("CALMAR_RATIO", Numeric)
    avg_trade_return: Mapped[Optional[float]] = mapped_column("AVG_TRADE_RETURN", Numeric)
    win_rate: Mapped[Optional[float]] = mapped_column("WIN_RATE", Numeric)
    profit_factor: Mapped[Optional[float]] = mapped_column("PROFIT_FACTOR", Numeric)
    risk_reward_win: Mapped[Optional[float]] = mapped_column("RISK_REWARD_WIN", Numeric)
    risk_reward_loss: Mapped[Optional[float]] = mapped_column("RISK_REWARD_LOSS", Numeric)
    avg_gain: Mapped[Optional[float]] = mapped_column("AVG_GAIN", Numeric)
    avg_risk: Mapped[Optional[float]] = mapped_column("AVG_RISK", Numeric)
    log_filename: Mapped[Optional[str]] = mapped_column("LOG_FILENAME", String)


class Dataset(Base):
    __tablename__ = "DATASET"
    sk_dataset: Mapped[int] = mapped_column("SK_DATASET", Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column("NAME", String)
    start_date: Mapped[Optional[datetime]] = mapped_column("START_DATE", DateTime)
    end_date: Mapped[Optional[datetime]] = mapped_column("END_DATE", DateTime)
    file_name: Mapped[Optional[str]] = mapped_column("FILE_NAME", String)
    description: Mapped[Optional[str]] = mapped_column("DESCRIPTION", String)
    
class Indicator(Base):
    __tablename__ = "INDICATOR"
    sk_indicator: Mapped[int] = mapped_column("SK_INDICATOR", Integer, primary_key=True, autoincrement=True)
    name: Mapped[Optional[str]] = mapped_column("NAME", String)
    label: Mapped[Optional[str]] = mapped_column("LABEL", String)
    code: Mapped[Optional[str]] = mapped_column("CODE", String)
    type: Mapped[Optional[int]] = mapped_column("TYPE", Integer)
