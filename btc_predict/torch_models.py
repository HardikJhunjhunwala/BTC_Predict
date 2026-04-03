from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def get_best_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TorchRegressor:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: torch.device
    loss_history: list[float]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        x = X.to_numpy(dtype=np.float32)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(xt).squeeze(-1).detach().cpu().numpy()
        return pred


@dataclass
class TorchClassifier:
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    device: torch.device
    loss_history: list[float]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        x = X.to_numpy(dtype=np.float32)
        x = (x - self.mean) / self.std
        xt = torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(xt).squeeze(-1)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        return probs

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int64)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _standardize_fit(X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = X.to_numpy(dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    return x, mean, std


def train_torch_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    batch_size: int,
    epochs: int,
    lr: float,
    show_progress: bool = True,
) -> TorchRegressor:
    device = get_best_device()
    x_np, mean, std = _standardize_fit(X_train)
    y_np = y_train.to_numpy(dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLPRegressor(in_dim=x_np.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loss_history: list[float] = []
    epbar = tqdm(range(epochs), desc=f"Train torch reg ({device.type})", disable=not show_progress)
    for _ in epbar:
        model.train()
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu().item()) * xb.size(0)
            n += xb.size(0)
        epoch_loss = running / max(n, 1)
        loss_history.append(epoch_loss)
        epbar.set_postfix(loss=f"{epoch_loss:.6f}")

    return TorchRegressor(model=model, mean=mean, std=std, device=device, loss_history=loss_history)


def train_torch_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    batch_size: int,
    epochs: int,
    lr: float,
    show_progress: bool = True,
) -> TorchClassifier:
    device = get_best_device()
    x_np, mean, std = _standardize_fit(X_train)
    y_np = y_train.to_numpy(dtype=np.float32)

    ds = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = MLPClassifier(in_dim=x_np.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    loss_history: list[float] = []
    epbar = tqdm(range(epochs), desc=f"Train torch cls ({device.type})", disable=not show_progress)
    for _ in epbar:
        model.train()
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu().item()) * xb.size(0)
            n += xb.size(0)
        epoch_loss = running / max(n, 1)
        loss_history.append(epoch_loss)
        epbar.set_postfix(loss=f"{epoch_loss:.6f}")

    return TorchClassifier(model=model, mean=mean, std=std, device=device, loss_history=loss_history)


def eval_torch_regression(model: TorchRegressor, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    rmse = mean_squared_error(y, pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y, pred),
        "RMSE": rmse,
        "R2": r2_score(y, pred),
    }


def eval_torch_classification(model: TorchClassifier, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, proba),
    }
