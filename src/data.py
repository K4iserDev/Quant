import pandas as pd



def load_mt5_csv(path):
    """
    En esta funcion estamos leyendo el contenido del csv y pasandolo con data y time juntos sin vol 
    """
    df = pd.read_csv(path, sep="\t")
    df.columns = df.columns.str.strip("<>").str.lower()
    texto = df["date"] + " " + df["time"]
    df["datetime"] = pd.to_datetime(texto, format="%Y.%m.%d %H:%M:%S")
    df = df.drop(columns=["date", "time", "vol"])
    df = df.set_index("datetime")
    df = df.sort_index()
    return df



