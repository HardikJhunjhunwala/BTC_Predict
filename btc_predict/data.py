import subprocess
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from btc_predict.config import Config

FEATURE_COLUMNS: List[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "hour", "dayofweek", "month",
    "ret_1", "ret_5", "ret_15", "ret_60",
    "close_sma_5", "close_sma_15", "close_sma_60", "close_sma_240",
    "close_std_5", "close_std_15", "close_std_60", "close_std_240",
    "vol_sma_5", "vol_sma_15", "vol_sma_60", "vol_sma_240",
    "ema_12", "ema_26", "macd",
    "rsi_14",
    "bb_upper", "bb_lower", "bb_width",
    "hl_range", "oc_change",
]


def _approx_data_rows(path: str) -> Optional[int]:
    try:
        out = subprocess.check_output(["wc", "-l", path], text=True)
        total_lines = int(out.split()[0])
        return max(0, total_lines - 1)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError, OSError):
        return None


def load_data(
    path: str,
    chunk_rows: int = 200_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    expected = {"Timestamp", "Open", "High", "Low", "Close", "Volume"}
    data_rows = _approx_data_rows(path)
    total_chunks = None
    if data_rows is not None and chunk_rows > 0:
        total_chunks = (data_rows + chunk_rows - 1) // chunk_rows

    chunks: List[pd.DataFrame] = []
    reader = pd.read_csv(path, chunksize=chunk_rows)
    iterator = tqdm(
        reader,
        desc="Loading CSV",
        total=total_chunks,
        unit="chunk",
        disable=not show_progress,
    )
    first = True
    for chunk in iterator:
        if first:
            missing = expected - set(chunk.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            first = False
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["Timestamp"], unit="s", errors="coerce")
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    return out


def add_technical_features(df: pd.DataFrame, show_progress: bool = True) -> pd.DataFrame:
    out = df.copy()

    out["ret_1"] = out["Close"].pct_change(1)
    out["ret_5"] = out["Close"].pct_change(5)
    out["ret_15"] = out["Close"].pct_change(15)
    out["ret_60"] = out["Close"].pct_change(60)

    windows = [5, 15, 60, 240]
    win_iter = tqdm(windows, desc="Rolling windows", disable=not show_progress, leave=False)
    for w in win_iter:
        out[f"close_sma_{w}"] = out["Close"].rolling(w).mean()
        out[f"close_std_{w}"] = out["Close"].rolling(w).std()
        out[f"vol_sma_{w}"] = out["Volume"].rolling(w).mean()

    out["ema_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["macd"] = out["ema_12"] - out["ema_26"]

    delta = out["Close"].diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up).rolling(14).mean()
    roll_down = pd.Series(down).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ma20 = out["Close"].rolling(20).mean()
    std20 = out["Close"].rolling(20).std()
    out["bb_upper"] = ma20 + 2 * std20
    out["bb_lower"] = ma20 - 2 * std20
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / (ma20 + 1e-9)

    out["hl_range"] = (out["High"] - out["Low"]) / (out["Close"] + 1e-9)
    out["oc_change"] = (out["Close"] - out["Open"]) / (out["Open"] + 1e-9)

    return out


def make_targets(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = df.copy()
    out["target_close_reg"] = out["Close"].shift(-horizon)
    out["target_return"] = (out["target_close_reg"] - out["Close"]) / (out["Close"] + 1e-9)
    out["target_direction_cls"] = (out["target_return"] > 0).astype(int)
    return out


def prepare_dataset(cfg: Config) -> Tuple[pd.DataFrame, List[str]]:
    df = load_data(
        cfg.csv_path,
        chunk_rows=cfg.csv_chunk_rows,
        show_progress=cfg.show_progress,
    )

    if cfg.sample_every_n_rows > 1:
        df = df.iloc[::cfg.sample_every_n_rows].copy().reset_index(drop=True)

    sp = cfg.show_progress
    pipeline = [
        ("Calendar features", lambda d: add_time_features(d)),
        ("Technical indicators", lambda d: add_technical_features(d, show_progress=sp)),
        (f"Targets (+{cfg.horizon_minutes}m)", lambda d: make_targets(d, cfg.horizon_minutes)),
    ]
    pbar = tqdm(pipeline, desc="Feature pipeline", disable=not sp)
    for label, fn in pbar:
        pbar.set_description(f"Features: {label}")
        df = fn(df)

    keep_cols = FEATURE_COLUMNS + [
        "datetime", "Close", "target_close_reg", "target_direction_cls", "target_return",
    ]
    df = df[keep_cols].dropna().reset_index(drop=True)

    return df, FEATURE_COLUMNS


def time_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    test_start = int(n * (1 - test_ratio))
    train_val = df.iloc[:test_start].copy()
    test = df.iloc[test_start:].copy()

    n_tv = len(train_val)
    val_start = int(n_tv * (1 - val_ratio))
    train = train_val.iloc[:val_start].copy()
    val = train_val.iloc[val_start:].copy()

    return train, val, test
