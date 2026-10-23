import numpy as np
import pandas as pd


def simular_trade(o, h, l, c, i, direccion, sl_dist, tp_dist, max_velas, coste):
    """
    Simula un trade que entra en la apertura de la vela i.
    o, h, l, c: arrays de NumPy con open, high, low, close.
    direccion: +1 compra, -1 venta. sl_dist/tp_dist y coste en precio ($).
    Si en una vela se tocan SL y TP, cuenta el SL (conservador).
    Devuelve un dict con el resultado del trade (pnl en $ y R).
    """
    entrada = o[i]  # ← CAMBIO: antes df["open"].iloc[i]
    Sl = entrada - direccion * sl_dist
    Tp = entrada + direccion * tp_dist
    motive = "time"
    output = None
    for j in range(i, i + max_velas):
        hi = h[j]  # ← CAMBIO: antes df["high"].iloc[j]
        lo = l[j]  # ← CAMBIO: antes df["low"].iloc[j]

        if direccion == 1:  # compra: SL abajo, TP arriba
            toca_sl = lo <= Sl
            toca_tp = hi >= Tp
        else:  # venta: SL arriba, TP abajo
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
        output = c[j]  # ← CAMBIO: antes df["close"].iloc[j]
    pnl = direccion * (output - entrada) - coste
    R = pnl / sl_dist
    return {
        "entrada_idx": i,
        "direccion": direccion,
        "entrada": entrada,
        "salida": output,
        "motivo": motive,
        "velas": j - i + 1,
        "pnl": pnl,
        "R": R,
    }


def backtest(df, señales, sl_mult, tp_mult, max_velas, coste):
    """
    Ejecuta todas las señales (+1/-1) con SL/TP en múltiplos de ATR.
    Requiere df["atr"]. Entra en la vela siguiente a la señal; una posición a la vez.
    Devuelve un DataFrame con un trade por fila.
    """
    o, h, l, c = (
        df[col].values for col in ["open", "high", "low", "close"]
    )  # arrays, una sola vez
    a_arr = df["atr"].values
    s_arr = señales.values  # ← CAMBIO: señales también como array

    idx_señales = np.flatnonzero(s_arr != 0)
    trades = []
    libre_desde = 0

    for idx in idx_señales:
        entrada = idx + 1
        a = a_arr[idx]  # ← CAMBIO: antes df["atr"].iloc[idx]
        if np.isnan(a) or a <= 0:
            continue
        if entrada < libre_desde:
            continue
        if entrada + max_velas > len(df):
            break
        sl_dist = sl_mult * a
        tp_dist = tp_mult * a
        res = simular_trade(
            o,
            h,
            l,
            c,
            entrada,
            int(s_arr[idx]),  # ← CAMBIO: arrays en vez de df
            sl_dist,
            tp_dist,
            max_velas,
            coste,
        )
        trades.append(res)
        libre_desde = entrada + res["velas"]

    return pd.DataFrame(trades)

def placebo_direccion(df, señales, n_sim=200, **kw):
    """Mismas velas de entrada que 'señales' con dirección aleatoria. Devuelve la expR de cada simulación."""
    mask = señales != 0
    res = []
    for seed in range(n_sim):
        rng = np.random.default_rng(seed)
        s = señales.copy()
        s[mask] = rng.choice([1, -1], size=mask.sum())
        res.append(backtest(df, s, **kw)["R"].mean())
    return np.array(res)