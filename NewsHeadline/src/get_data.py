import sqlalchemy
import pandas as pd

def connect(user, password, db, host: str, port: int, echo=False):
    url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)
    eng = sqlalchemy.create_engine(url, client_encoding='utf8', echo=echo)
    meta = sqlalchemy.MetaData(bind=eng)
    return eng, meta

def get_data_frame(pair: str = 'USDT_BTC', exchange: str = 'poloniex') -> pd.DataFrame:
    engine, meta = connect(user='postgres', password='password', db='btccandles', \
                                            host='94.230.125.199', port=16432)
    df = pd.read_sql_query(
        "SELECT open,close,low,high,volume,date,adj_close,exchange.name AS exchange,pair.name AS pair "
         "FROM candlestick, exchange, pair "
         "WHERE candlestick.exchange_id = exchange.id "
         "AND candlestick.pair_id = pair.id AND exchange.name = '"\
             + exchange + "' AND pair.name = '" + pair +
         "' ORDER BY candlestick.date, candlestick.time"
        ,con=engine)
    return df

def get_data_frame5minutes(pair: str = 'USDT_BTC', exchange: str = 'poloniex') -> pd.DataFrame:
    engine, meta = connect(user='postgres', password='password', db='data_dev5', host='94.230.125.199', port=16432)
    df = pd.read_sql_query(
        "SELECT open,close,low,high,volume,date,time,adj_close,exchange.name AS exchange,pair.name AS pair "
            "FROM candlestick, exchange, pair "
            "WHERE candlestick.exchange_id = exchange.id "
            "AND candlestick.pair_id = pair.id AND exchange.name = '" + exchange + "' AND pair.name = '" + pair + 
            "' ORDER BY candlestick.date, candlestick.time"
        ,con=engine)
    return df


def get_specific_data_frame(name, owner) -> pd.DataFrame:
    
        df = get_data_frame(name,owner)
        return df

def loadHeadlines():
    engine, meta = connect(user='postgres', password='password', \
                           db='btccandles', host='94.230.125.199', port=16432)
    df = pd.read_sql_query(
        "SELECT * from parsing_news ORDER BY date DESC , time DESC", con =engine)

    df.date = pd.to_datetime(df['date'])
    df.index = df.date
    return df