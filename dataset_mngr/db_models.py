from datetime import datetime
from sqlalchemy import ForeignKey, Integer, String
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


class SymbolInfo(Base):
    __tablename__ = "SYMBOL_INFO"

    sk_symbol_info: Mapped[int] = mapped_column(
        primary_key=True, autoincrement=True)
    SK_SYMBOL: Mapped[int] = mapped_column(ForeignKey("Symbol.sk_symbol"))
    info : Mapped[str]
    update_date : Mapped[datetime]
    active_row : Mapped[Optional[int]]
    info_clean : Mapped[Optional[str]]
