import threading
import time
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

warnings.filterwarnings("ignore")

_RF_TYPES = (RandomForestRegressor, RandomForestClassifier)


def _fit_model_with_progress(
    name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    show_progress: bool,
    heartbeat_sec: float = 15.0,
) -> None:
    if not show_progress:
        model.fit(X_train, y_train)
        return

    long_job = isinstance(model, _RF_TYPES)
    stop = threading.Event()

    def heartbeat():
        t0 = time.perf_counter()
        while not stop.wait(heartbeat_sec):
            tqdm.write(
                f"    … {name}: still training ({time.perf_counter() - t0:,.0f}s elapsed, "
                "no sub-steps in sklearn RF)"
            )

    hb_thread = None
    if long_job:
        hb_thread = threading.Thread(target=heartbeat, daemon=True)
        hb_thread.start()

    pbar = tqdm(
        total=1,
        desc=f"Train: {name}",
        bar_format="{l_bar}{bar}| {desc} [{elapsed}]",
        leave=True,
    )
    try:
        model.fit(X_train, y_train)
    finally:
        stop.set()
        if hb_thread is not None:
            hb_thread.join(timeout=1.0)
        pbar.update(1)
        pbar.close()


def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    show_progress: bool = True,
) -> Dict[str, object]:
    models = {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        _fit_model_with_progress(name, model, X_train, y_train, show_progress)
    return models


def train_classification_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    show_progress: bool = True,
) -> Dict[str, object]:
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000)),
        ]),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=250,
            max_depth=14,
            random_state=42,
            n_jobs=-1,
        ),
    }

    for name, model in models.items():
        _fit_model_with_progress(name, model, X_train, y_train, show_progress)
    return models


def eval_regression(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": rmse,
        "R2": r2_score(y, pred),
    }


def eval_classification(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    else:
        auc = np.nan

    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": auc,
    }
