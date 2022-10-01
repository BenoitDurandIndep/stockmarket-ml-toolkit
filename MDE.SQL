SELECT sk_symbol,COUNT(*) FROM candle GROUP BY sk_symbol;
--TRUNCATE TABLE candle;