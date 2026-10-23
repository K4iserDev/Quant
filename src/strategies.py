import pandas as pd
import numpy as np


def señales_ruptura_ny(df, compresion=None, ventana_dias=100):
    """
    Estrategia B: ruptura del rango 14:00–15:25 durante 15:30–16:55 (hora servidor). 1 señal/día.
    compresion: si es un número (p. ej. 0.3), solo opera los días cuyo rango (en ATR) está por debajo
                de ese percentil de los 'ventana_dias' días anteriores. Requiere df['atr'].
    """
    fecha = df.index.date
    hm = df.index.hour * 100 + df.index.minute

    en_rango = (hm >= 1400) & (hm <= 1525)
    rango = df[en_rango].groupby(fecha[en_rango]).agg(r_high=("high", "max"),
                                                      r_low=("low", "min"),
                                                      atr_fin=("atr", "last"))   # ATR al final del rango

    r_high = pd.Series(fecha, index=df.index).map(rango["r_high"])
    r_low = pd.Series(fecha, index=df.index).map(rango["r_low"])

    en_ventana = (hm >= 1530) & (hm <= 1655)
    compra = en_ventana & (df["close"] > r_high)
    venta = en_ventana & (df["close"] < r_low)

    # --- Filtro de compresión (B1) ---
    if compresion is not None:
        anchura = (rango["r_high"] - rango["r_low"]) / rango["atr_fin"]           # anchura del rango en ATRs
        umbral = anchura.rolling(ventana_dias).quantile(compresion).shift(1)      # percentil de los días ANTERIORES
        dia_ok = anchura < umbral                                                 # True = rango comprimido
        ok = pd.Series(fecha, index=df.index).map(dia_ok).fillna(False).astype(bool)
        compra = compra & ok
        venta = venta & ok

    señal = pd.Series(0, index=df.index)
    señal[compra] = 1
    señal[venta] = -1

    n = (señal != 0).groupby(fecha).cumsum()
    señal[n > 1] = 0
    return señal

def señales_reversion(df, horas=(11, 22), k=1.5, n_ema=48):
    """Estrategia A: reversión a la EMA en las horas dadas cuando |z| > k (z en ATRs). Requiere df['atr']."""
    fecha = df.index.date
    hora = df.index.hour

    ema = df["close"].ewm(span=n_ema, adjust=False).mean()
    atr_ok = df["atr"].replace(0, np.nan)
    z = (df["close"] - ema) / atr_ok
    

    en_hora = np.isin(hora, horas)
    compra = en_hora & (z < -k)
    venta = en_hora & (z > k)

    señal = pd.Series(0, index=df.index)
    señal[compra] = 1
    señal[venta] = -1

    n = (señal != 0).groupby([fecha, hora]).cumsum()
    señal[n > 1] = 0
    return señal

