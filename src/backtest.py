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
