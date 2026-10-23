import numpy as np
import pandas as pd

from src.backtest import simular_trade


def crear_features(df, coste=0.25):
    """
    Features causales (solo usan datos hasta el cierre de cada vela) para todas las velas.
    Requiere df['atr']. Devuelve un DataFrame con el mismo índice que df.
    """
    c, a = df["close"], df["atr"].replace(0, np.nan)  # ATR=0 → NaN (velas planas)
    ema48 = c.ewm(span=48, adjust=False).mean()
    ema288 = c.ewm(span=288, adjust=False).mean()
    hora = df.index.hour

    f = pd.DataFrame(index=df.index)
    f["z"] = (c - ema48) / a  # muelle de 4 h
    f["z_dia"] = (c - ema288) / a  # desviación respecto al día
    f["pend_dia"] = (ema288 - ema288.shift(12)) / a  # tendencia del día (última hora)
    f["atr_ratio"] = (
        a / a.rolling(288 * 5).mean()
    )  # régimen de volatilidad (≈ 1 semana)
    for n in [1, 3, 12]:
        f[f"ret_{n}"] = (c - c.shift(n)) / a  # movimiento reciente en ATR
    f["rango_12"] = (df["high"].rolling(12).max() - df["low"].rolling(12).min()) / a
    f["coste_rel"] = coste / a  # peso del coste
    f["hora_sin"] = np.sin(2 * np.pi * hora / 24)  # hora cíclica
    f["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    f["dia_sem"] = df.index.dayofweek  # 0 = lunes
    return f


def crear_dataset(df, señales, features, sl_mult, tp_mult, max_velas, coste):
    """
    Una fila por señal: features en la vela de la señal + dirección + resultado real del trade.
    Cada señal se simula de forma independiente (sin la regla de 1 posición), para etiquetar.
    """
    o, h, l, c = (df[col].values for col in ["open", "high", "low", "close"])
    a_arr = df["atr"].values
    s_arr = señales.values

    filas = []
    for idx in np.flatnonzero(s_arr != 0):
        entrada = idx + 1
        a = a_arr[idx]
        if np.isnan(a) or a <= 0 or entrada + max_velas > len(df):
            continue
        res = simular_trade(
            o,
            h,
            l,
            c,
            entrada,
            int(s_arr[idx]),
            sl_mult * a,
            tp_mult * a,
            max_velas,
            coste,
        )
        filas.append({"idx": idx, "direccion": int(s_arr[idx]), "R": res["R"]})

    ds = pd.DataFrame(filas)
    ds = ds.join(
        features.iloc[ds["idx"]].reset_index()
    )  # features de la vela de la señal
    ds["y"] = (ds["R"] > 0).astype(int)  # etiqueta: ¿trade ganador?
    return ds.dropna()
