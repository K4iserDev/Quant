import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def half_life(x):
    """Half-life (en velas) de un proceso OU estimado con Δx_t = a + b·x_{t-1}."""
    x = x.dropna().values
    dx = np.diff(x)  # Δx_t
    b = np.polyfit(x[:-1], dx, 1)[0]  # pendiente de la regresión
    return -np.log(2) / np.log(1 + b)


def hurst(precio_log, max_lag=100):
    """Exponente de Hurst: pendiente de log(dispersión) frente a log(lag)."""
    x = precio_log.values
    lags = np.arange(2, max_lag)
    tau = [
        np.std(x[lag:] - x[:-lag]) for lag in lags
    ]  # dispersión del cambio de precio a cada lag
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


def acf_rapida(serie, n_lags=20, titulo=""):
    """Calcula la ACF lag a lag con pandas y la pinta con la banda de ruido al 95 %."""
    serie = serie.dropna()
    lags = range(1, n_lags + 1)
    acf = pd.Series(
        [serie.autocorr(lag=l) for l in lags], index=lags
    )  # autocorrelación de cada lag
    banda = 1.96 / np.sqrt(len(serie))  # límite del ruido

    acf.plot(kind="bar", figsize=(10, 4))
    plt.axhline(banda, color="red", ls="--")
    plt.axhline(-banda, color="red", ls="--")
    plt.axhline(0, color="black", lw=0.8)
    plt.title(titulo)
    plt.xlabel("lag (velas)")
    plt.ylabel("autocorrelación")
    plt.show()
    return acf
