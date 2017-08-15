import numpy as np
import pandas as pd

def wilder_smooth(value_list, period):
    nan_offset = np.isnan(value_list).sum()
    smoothened = [np.nan for i in range(period-1+nan_offset)]
    smoothened.append(np.mean(value_list[nan_offset:period+nan_offset]))
    for i in range(period+nan_offset, len(value_list)):
        smoothened.append(smoothened[i-1] + (value_list[i] - smoothened[i-1]) / period)
    return np.array(smoothened)


def ma_rel_diff(df, period=50):
    df['ma_rel_diff_{}'.format(period)] = 1 - df['close'].rolling(period).mean().values / df['close']
    return df


def ema_rel_diff(df, period=10):
    df['ema_rel_diff_{}'.format(period)] = 1 - df['close'].ewm(span=period, min_periods=period-1).mean().values / df['close']
    return df


def mom(df, period=20):
    df['moment_{}'.format(period)] = df['close'].diff(period).values
    return df


def roc(df, period=14):
    df['roc_{}'.format(period)] = df['close'] / df['close'].shift(period).values
    return df


def bbands(df, period=20, std_off=2):
    boil_mean = df['close'].rolling(period).mean().to_frame(name='boil_mean_{}_{}'.format(period, std_off))
    boil_std = df['close'].rolling(period).std().to_frame(name='boil_std_{}_{}'.format(period, std_off))

    boil_up = (boil_mean['boil_mean_{}_{}'.format(period, std_off)] + std_off*boil_std['boil_std_{}_{}'.format(period, std_off)]).to_frame(name='boil_up_{}_{}'.format(period, std_off))
    boil_down = (boil_mean['boil_mean_{}_{}'.format(period, std_off)] - std_off*boil_std['boil_std_{}_{}'.format(period, std_off)]).to_frame(name='boil_down_{}_{}'.format(period, std_off))

    df = df.join(boil_mean)
    df = df.join(boil_up)
    df = df.join(boil_down)

    return df

def normalized_bbands(df, period=20, std_off=20):
    boil_mean = df['close'].rolling(period).mean()
    boil_std = df['close'].rolling(period).std()

    boil_up = df['close'].values / (boil_mean + std_off*boil_std) - 1
    boil_down = df['close'].values / (boil_mean - std_off*boil_std) - 1

    boil_up = boil_up * boil_up.gt(0)
    boil_down = boil_down * boil_down.lt(0)

    df['normBB'] = boil_up.values + boil_down.values

    return df


def rsi(df, period=14):
    df['rsi_{}'.format(period)] = 100.0 - 100.0 / (1.0 + df['close'].diff(1).gt(0).rolling(period).mean().values / df['close'].diff(1).lt(0).rolling(period).mean().values)
    return df


def stochastics(df, period=14, smooth=3):
    stoch_k = (100.0 * (df['close'] - df['low'].rolling(period).min()) / (df['high'].rolling(period).max() - df['low'].rolling(period).min())).to_frame(name='stoch_k_{}'.format(period))
    stoch_d = stoch_k['stoch_k_{}'.format(period)].rolling(smooth).mean().to_frame(name='stoch_d_{}_{}'.format(period, smooth))
    df = df.join(stoch_d)

    return df


def macd(df, period_fast=12, period_slow=26, smooth=9):
    macd = df['close'].ewm(span=period_fast, min_periods=period_fast-1).mean() - df['close'].ewm(span=period_slow, min_periods=period_slow-1).mean()
    macd_smoothed = macd.ewm(span=smooth, min_periods=smooth-1).mean().values

    df['macd_{}_{}_{}'.format(period_fast, period_slow, smooth)] = macd - macd_smoothed

    return df


def atr(df, period=14):
    TR = pd.concat([df['high'], df['close'].shift(1)], 1).max(1) - pd.concat([df['low'], df['close'].shift(1)], 1).min(1)
    ATR = TR.rolling(period).mean().to_frame(name='atr_{}'.format(period))
    return df.join(ATR)


def adx(df, period=14):
    TR = pd.concat([df['high'], df['close'].shift(1)], 1).max(1) - pd.concat([df['low'], df['close'].shift(1)], 1).min(1)
    df['ATR'] = wilder_smooth(TR, period)

    up_down = df['high'].diff(1).gt(-1*df['low'].diff(1))

    pDM = df['high'].diff(1) * df['high'].diff(1).gt(0) * up_down
    mDM = df['low'].diff(1) * df['low'].diff(1).lt(0) * (up_down - 1)

    pDI = 100 * wilder_smooth(pDM.values, period) / df['ATR']
    mDI = 100 * wilder_smooth(mDM.values, period) / df['ATR']

    DX = (100 * (pDI - mDI).abs() / (pDI + mDI))

    ADX = pd.DataFrame(wilder_smooth(DX, period), index=df.index, columns=['adx_{}'.format(period)])

    return df.join(ADX)

def volatility(df):
    volatility = pd.DataFrame(df.close).rolling(WINDOW).std().values.tolist()
    volatility = [v[0] for v in volatility]
    v = pd.DataFrame(data = volatility, columns = ['volatility'])
    return df.join(v)
