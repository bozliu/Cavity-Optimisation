"""Model definitions and metric utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from .device import get_torch_device


@dataclass
class Candidate:
    name: str
    family: str
    builder: Callable[[], Any]


def evaluate_radius_metrics(y_true: np.ndarray, y_pred_class: np.ndarray) -> dict[str, float]:
    exact = float(accuracy_score(y_true, y_pred_class))
    within_1 = float((np.abs(y_true - y_pred_class) <= 1).mean())
    return {"radius_accuracy": exact, "radius_within_1_class": within_1}


def evaluate_height_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "height_r2": float(r2_score(y_true, y_pred)),
        "height_mae": float(mean_absolute_error(y_true, y_pred)),
    }


class TorchRadiusClassifier:
    """Lightweight feedforward classifier for cR classes."""

    def __init__(
        self,
        hidden: list[int],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        random_state: int,
        num_classes: int,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.num_classes = num_classes
        self.device = get_torch_device()

        self.model: torch.nn.Module | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _build_model(self, n_features: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        in_dim = n_features
        for hidden_dim in self.hidden:
            layers.extend([torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(), torch.nn.BatchNorm1d(hidden_dim)])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, self.num_classes))
        return torch.nn.Sequential(*layers)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model is not fitted")
        return (x - self.mean_) / self.std_

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "TorchRadiusClassifier":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.mean_ = x_train.mean(axis=0, keepdims=True)
        self.std_ = x_train.std(axis=0, keepdims=True) + 1e-8

        x_train_std = self._standardize(x_train).astype(np.float32)
        train_x = torch.tensor(x_train_std, dtype=torch.float32)
        train_y = torch.tensor(y_train.astype(np.int64), dtype=torch.long)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model(n_features=x_train.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.CrossEntropyLoss()

        best_state = None
        best_val_acc = -1.0
        patience = 50
        stale = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            if x_val is not None and y_val is not None:
                pred = self.predict(x_val)
                val_acc = float((pred == y_val).mean())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        x_std = self._standardize(x).astype(np.float32)
        with torch.no_grad():
            logits = self.model(torch.tensor(x_std, dtype=torch.float32, device=self.device))
            pred = logits.argmax(dim=1).cpu().numpy()
        return pred


class TorchHeightRegressor:
    """Lightweight feedforward regressor for cH."""

    def __init__(
        self,
        hidden: list[int],
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        random_state: int,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = get_torch_device()

        self.model: torch.nn.Module | None = None
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def _build_model(self, n_features: int) -> torch.nn.Module:
        layers: list[torch.nn.Module] = []
        in_dim = n_features
        for hidden_dim in self.hidden:
            layers.extend([torch.nn.Linear(in_dim, hidden_dim), torch.nn.ReLU(), torch.nn.BatchNorm1d(hidden_dim)])
            in_dim = hidden_dim
        layers.append(torch.nn.Linear(in_dim, 1))
        return torch.nn.Sequential(*layers)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Model is not fitted")
        return (x - self.mean_) / self.std_

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "TorchHeightRegressor":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.mean_ = x_train.mean(axis=0, keepdims=True)
        self.std_ = x_train.std(axis=0, keepdims=True) + 1e-8

        x_train_std = self._standardize(x_train).astype(np.float32)
        train_x = torch.tensor(x_train_std, dtype=torch.float32)
        train_y = torch.tensor(y_train.astype(np.float32), dtype=torch.float32).unsqueeze(1)

        dataset = torch.utils.data.TensorDataset(train_x, train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = self._build_model(n_features=x_train.shape[1]).to(self.device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss()

        best_state = None
        best_val_r2 = float("-inf")
        patience = 50
        stale = 0

        for _ in range(self.epochs):
            self.model.train()
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

            if x_val is not None and y_val is not None:
                pred = self.predict(x_val)
                val_r2 = float(r2_score(y_val, pred))
                if val_r2 > best_val_r2:
                    best_val_r2 = val_r2
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted")
        x_std = self._standardize(x).astype(np.float32)
        with torch.no_grad():
            pred = self.model(torch.tensor(x_std, dtype=torch.float32, device=self.device)).cpu().numpy().reshape(-1)
        return pred


def build_radius_candidates(config: dict[str, Any], num_classes: int) -> list[Candidate]:
    seed = int(config["random_state"])
    model_cfg = config["models"]
    torch_cfg = model_cfg["torch"]

    return [
        Candidate("decision_tree_classifier", "legacy", lambda: DecisionTreeClassifier(random_state=seed)),
        Candidate(
            "random_forest_classifier",
            "legacy",
            lambda: RandomForestClassifier(
                n_estimators=int(model_cfg["random_forest_n_estimators"]),
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "extra_trees_classifier",
            "modern",
            lambda: ExtraTreesClassifier(
                n_estimators=int(model_cfg["extra_trees_n_estimators"]),
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "torch_radius_classifier",
            "legacy_nn",
            lambda: TorchRadiusClassifier(
                hidden=list(torch_cfg["radius_hidden"]),
                epochs=int(torch_cfg["radius_epochs"]),
                batch_size=int(torch_cfg["batch_size"]),
                learning_rate=float(torch_cfg["learning_rate"]),
                weight_decay=float(torch_cfg["weight_decay"]),
                random_state=seed,
                num_classes=num_classes,
            ),
        ),
    ]


def build_height_candidates(config: dict[str, Any]) -> list[Candidate]:
    seed = int(config["random_state"])
    model_cfg = config["models"]
    torch_cfg = model_cfg["torch"]

    return [
        Candidate("linear_regression", "legacy", lambda: make_pipeline(StandardScaler(), LinearRegression())),
        Candidate(
            "knn_regressor",
            "legacy",
            lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=int(model_cfg["knn_neighbors"]))),
        ),
        Candidate("decision_tree_regressor", "legacy", lambda: DecisionTreeRegressor(random_state=seed)),
        Candidate(
            "extra_trees_regressor",
            "modern",
            lambda: ExtraTreesRegressor(
                n_estimators=int(model_cfg["extra_trees_n_estimators"]),
                random_state=seed,
                n_jobs=-1,
            ),
        ),
        Candidate(
            "torch_height_regressor",
            "legacy_nn",
            lambda: TorchHeightRegressor(
                hidden=list(torch_cfg["height_hidden"]),
                epochs=int(torch_cfg["height_epochs"]),
                batch_size=int(torch_cfg["batch_size"]),
                learning_rate=float(torch_cfg["learning_rate"]),
                weight_decay=float(torch_cfg["weight_decay"]),
                random_state=seed,
            ),
        ),
    ]


def build_multioutput_candidate(config: dict[str, Any]) -> Candidate:
    seed = int(config["random_state"])
    model_cfg = config["models"]

    return Candidate(
        "multioutput_extra_trees",
        "modern",
        lambda: MultiOutputRegressor(
            ExtraTreesRegressor(
                n_estimators=int(model_cfg["multioutput_extra_trees_n_estimators"]),
                random_state=seed,
                n_jobs=-1,
            )
        ),
    )
