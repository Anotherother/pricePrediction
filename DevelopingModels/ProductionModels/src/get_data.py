import sqlalchemy
import pandas as pd


def connect(user, password, db, host: str, port: int, echo=False):
    url = 'postgresql+psycopg2://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)
    eng = sqlalchemy.create_engine(url, client_encoding='utf8', echo=echo)
    meta = sqlalchemy.MetaData(bind=eng)

    return eng, meta

def get_data_frame(pair: str = 'USDT_BTC', exchange: str = 'poloniex') -> pd.DataFrame:

    engine, meta = connect(user='postgres', password='password', db='btc_dev', host='94.230.125.199', port=16432)
    df = pd.read_sql_query(
        "SELECT date, time, open, close, low, high, volume, pair.\"name\""
        "FROM candlestick, pair, exchange WHERE candlestick.exchange_id = exchange.id "
        "AND pair.id = candlestick.pair_id AND exchange.name = '" + exchange + "' AND candlestick.pair_id IN (SELECT pair.id FROM pair "
        "WHERE pair.alias_id = (SELECT pair.id FROM pair WHERE pair.name = '" + pair + "') OR pair.name = '" + pair + "');",con=engine)
    return df
