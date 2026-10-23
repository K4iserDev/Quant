import numpy as np
import pandas as pd 

def simular_trade(df, i, direccion, sl_dist, tp_dist, max_velas, coste):
    entrada = df["open"].iloc[i]
    Sl = entrada - direccion * sl_dist
    Tp = entrada + direccion * tp_dist
    motive = "time"
    output = None
    for j in range(i, i + max_velas):
        hi = df["high"].iloc[j]
        lo = df["low"].iloc[j]

        if direccion == 1:  # compra: SL abajo, TP arriba
            toca_sl = lo <= Sl
            toca_tp = hi >= Tp
        else:
            toca_sl = hi >= Sl
            toca_tp = lo <= Tp
        if toca_sl:
            output = Sl
            motive = "SL"
            break
        elif toca_tp:
            output = Tp
            motive = "TP"
            break
    if motive == "time":
        output = df["close"].iloc[j]
    pnl =  direccion * (output-entrada)-coste
    R = pnl / sl_dist
    return {"entrada_idx": i,
        "direccion": direccion,
        "entrada": entrada,
        "salida": output,
        "motivo": motive,
        "velas": j - i + 1,
        "pnl":pnl,
        "R": R}


def backtest(df, señales, sl_dist, tp_dist, max_velas, coste):
    idx_señales = np.flatnonzero(señales.values != 0)
    trades = []
    libre_desde = 0

    for idx in idx_señales:
        entrada = idx +1
        if entrada < libre_desde:
            continue
        if entrada + max_velas > len(df):
            break
        res = simular_trade(df, entrada,señales.iloc[idx], sl_dist, tp_dist, max_velas, coste)
        trades.append(res)
        libre_desde = entrada + res["velas"]

    return pd.DataFrame(trades)
    """
    señales: Serie con el mismo índice que df → +1 compra, -1 venta, 0 nada
    Devuelve un DataFrame con un trade por fila.
    """