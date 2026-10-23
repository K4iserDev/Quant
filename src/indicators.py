import pandas as pd


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    c1 = df["high"] - df["low"]
    c2 = (df["high"] - prev_close).abs()
    c3 = (df["low"] - prev_close).abs()
    tr = pd.concat([c1, c2, c3], axis=1).max(axis=1)
    return tr.rolling(n).mean()
