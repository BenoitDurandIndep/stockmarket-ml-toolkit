/********************/
/****** CANDLE ******/
/********************/
SELECT sk_symbol,COUNT(*),MIN(open_datetime),MAX(open_datetime) FROM candle GROUP BY sk_symbol;
SELECT sk_symbol,COUNT(*),MIN(open_datetime),MAX(open_datetime) FROM candle_save_23_10_08 GROUP BY sk_symbol;

SELECT OPEN_DATETIME, COUNT(*) FROM candle GROUP BY 1 HAVING COUNT(*)>1;

SELECT * FROM candle WHERE  sk_symbol=2 AND open_datetime>='2023-04-01' ORDER BY 1 ;

SELECT * FROM candle WHERE  sk_symbol=2 AND close IS NULL;
--DELETE FROM candle WHERE  sk_symbol=2 AND close IS NULL;

--DELETE FROM candle
WHERE SK_CANDLE NOT IN (
    SELECT MIN(SK_CANDLE)
    FROM candle WHERE SK_SYMBOL=2
    GROUP BY OPEN_DATETIME, TIMEFRAME
) AND SK_SYMBOL=2;

--UPDATE candle SET 
  OPEN=ROUND(OPEN,2),
    high=ROUND(high,2),
      low=ROUND(low,2),
  close=ROUND(close,2),
    adj_close=ROUND(adj_close,2);


SELECT open_datetime,TIMEFRAME, COUNT(*) AS nb FROM candle WHERE SK_SYMBOL=2 GROUP BY 1,2 having nb>1 ORDER BY 1;

/*TRUNCATE TABLE candle;*/

SELECT * FROM candle WHERE open_datetime BETWEEN '2020-02-05' AND '2020-03-20' AND sk_symbol=2 ORDER BY open_datetime;

SELECT CODE,OPEN_DATETIME, CLOSE FROM SYMBOL sym INNER JOIN CANDLE can ON sym.SK_SYMBOL=can.SK_SYMBOL 
    WHERE sym.CODE='CW8' AND can.TIMEFRAME=1440 ;

  /********************/
  /***** DATASET ******/
  /********************/
  /* DUPLICATE CONTENT DATASET */
--  insert into ds_content (sk_dataset,sk_symbol,sk_indicator)
SELECT 2,sk_symbol,sk_indicator FROM ds_content WHERE sk_dataset=1;

  SELECT ind.* FROM  dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR
  WHERE dts.NAME='DCA_CLOSE_1D_V1' AND sym.CODE='CW8' and ind.type=0;
  
  SELECT dts.SK_DATASET,sym.SK_SYMBOL,lab.SK_INDICATOR AS SK_LABEL,ind.SK_INDICATOR,ind.LABEL FROM dataset dts
  INNER JOIN ds_content dsc ON dts.SK_DATASET=dsc.SK_DATASET
  INNER JOIN symbol sym ON dsc.SK_SYMBOL=sym.SK_SYMBOL
  INNER JOIN indicator ind ON dsc.SK_INDICATOR=ind.SK_INDICATOR and ind.type=1
    INNER JOIN ds_content dsc_lab ON dts.SK_DATASET=dsc_lab.SK_DATASET
	 INNER JOIN indicator lab ON dsc_lab.SK_INDICATOR=lab.SK_INDICATOR AND lab.type=2
  WHERE dts.NAME='DCA_CLOSE_1D_V1' AND sym.CODE='CW8' AND lab.LABEL='lab_perf_21d' 
  AND ind.LABEL IN ('pos_bot_200','pos_sma200') ;
  
--    INSERT INTO ds_content (SK_DATASET,SK_SYMBOL,SK_INDICATOR) VALUES (1,2,31);
--  DELETE FROM ds_content where sk_dataset=2;
  
  /********************/
  /* MODEL DS_FILTERED /
  /********************/
--DELETE FROM model WHERE SK_MODEL=1;

--INSERT INTO model (SK_SYMBOL,SK_DATASET,SK_LABEL,ALGO,NAME,COMMENT)
SELECT distinct sym.SK_SYMBOL,dts.SK_DATASET,lab.SK_INDICATOR,'RANDOM FOREST CLASS' AS ALGO,
CONCAT(sym.code,'_',dts.NAME,'_',lab.label,'_','RANDOM_FOREST_CLASS') AS NAME,null AS COMMENT
FROM symbol sym
INNER JOIN ds_content con ON sym.SK_SYMBOL=con.SK_SYMBOL
INNER JOIN dataset dts ON con.SK_DATASET=dts.SK_DATASET
INNER JOIN indicator lab ON con.SK_INDICATOR=lab.SK_INDICATOR
WHERE sym.CODE='CW8' AND dts.NAME='DCA_CLOSE_1D_V1' AND lab.LABEL='lab_perf_21d';

--INSERT INTO ds_filtered (SK_MODEL,SK_INDICATOR)
SELECT DISTINCT md.SK_MODEL, ind.SK_INDICATOR FROM indicator ind
INNER JOIN model md
WHERE md.NAME='CW8_DCA_CLOSE_1D_V1_lab_perf_21d_LSTM_CLASS' 
AND  ind.label IN (
'pos_sma200',
'pos_sma50_200',
'pos_sma20_50',
'rsi14',
'sma20_rsi14',
'ret_1d',
'ret_5d',
'pos_top20',
'pos_top50',
'pos_bot20',
'pos_bot50',
'aroon14_up',
'macd_dif',
'pos_top_200',
'pos_bot_200',
'stdev20_sma5',
'pos_stdev20_sma5',
'stdev20_sma20',
'pos_rsi14_sma5',
'pos_rsi14_sma20',
'pos_bb20_hi',
 'pos_bb20_lo',
'stdev20_1d');

SELECT distinct ind.LABEL  FROM model md
INNER JOIN ds_filtered df ON md.SK_MODEL=df.SK_MODEL
INNER JOIN indicator ind ON df.SK_INDICATOR=ind.SK_INDICATOR
WHERE md.NAME='CW8_DCA_CLOSE_1D_V1_lab_perf_21d_RANDOM_FOREST_REG';

INSERT INTO ds_filtered(SK_MODEL,SK_INDICATOR)
SELECT 7, SK_INDICATOR FROM ds_filtered WHERE SK_MODEL=5; 

DELETE FROM ds_filtered WHERE sk_model IN (5,6,7) AND SK_INDICATOR=38;

DELETE FROM ds_filtered WHERE SK_MODEL IN (5,6,7) AND SK_INDICATOR IN (31,37,38,40,41);