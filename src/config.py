from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data/raw"
RAW_CSV = DATA_RAW / "XAUUSD_M5.csv"
DATA_PROCESSED = ROOT / "data/processed"
PROCESSED = DATA_PROCESSED / "xauusd_m5.parquet"
REPORTS = ROOT / "reports"
SYMBOL = "XAUUSD"
POINTS = 0.01
SPREAD_FLOOR = 20
COMMISION_POINTS = 0 #CapitalPointTrading-MT5
SLIPPAGE_POINTS = 5
