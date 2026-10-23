import pandas as pd
import numpy as np


def señales_ruptura_ny(df):
    fecha = df.index.date
    hm = df.index.hour * 100 + df.index.minute

    # 2. rango de cada día
    en_rango = (hm >= 1400) & (hm <= 1525)
    rango = (
        df[en_rango]
        .groupby(fecha[en_rango])
        .agg(r_high=("high", "max"), r_low=("low", "min"))
    )

    # 3. rango asignado a cada vela
    r_high = pd.Series(fecha, index=df.index).map(rango["r_high"])
    r_low = pd.Series(fecha, index=df.index).map(rango["r_low"])

    # 4. rupturas dentro de la ventana
    en_ventana = (hm >= 1530) & (hm <= 1655)
    compra = en_ventana & (df["close"] > r_high)
    venta = en_ventana & (df["close"] < r_low)

    # 5. señal
    señal = pd.Series(0, index=df.index)
    señal[compra] = 1
    señal[venta] = -1

    # 6. solo la primera del día
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

