SELECT sk_symbol,COUNT(*) FROM candle GROUP BY sk_symbol;
/*TRUNCATE TABLE candle;*/

SELECT * FROM candle WHERE open_datetime BETWEEN '2020-02-05' AND '2020-03-20' AND sk_symbol=2 ORDER BY open_datetime
;

SELECT CODE,OPEN_DATETIME, CLOSE FROM SYMBOL sym INNER JOIN CANDLE can ON sym.SK_SYMBOL=can.SK_SYMBOL 
    WHERE sym.CODE='CW8' AND can.TIMEFRAME=1440 ;
    
  SELECT ind.NAME,ind.LABEL,ind.CODE FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='DCA_CLOSE_1D_V1' AND sym.CODE='CW8';
  
  INSERT INTO ds_content (SK_DATASET,SK_SYMBOL,SK_INDICATOR) VALUES (1,2,31);
  
  SELECT * FROM candle WHERE sk_symbol=2 AND close IS NULL;
  
  DELETE FROM candle WHERE sk_symbol=2 AND close IS NULL;

  UPDATE candle SET 
  OPEN=ROUND(OPEN,2),
    high=ROUND(high,2),
      low=ROUND(low,2),
  close=ROUND(close,2),
    adj_close=ROUND(adj_close,2);
  
  