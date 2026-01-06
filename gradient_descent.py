#!/usr/bin/env python3
"""
Partie 1 — Régression logistique multi-classes (softmax) + descente de gradient (from-scratch)
Dataset : sklearn.datasets.load_digits (images 8x8 => 64 features)

Exécution :
    python gradient_descent.py
    python gradient_descent.py --alpha 0.3 --max_epochs 500 --eps 1e-5 --seed 42 --train_size 1437

Sorties :
    - results_part1_scratch.json
    - figures PNG (loss, matrice de confusion, poids 8x8, exemples mal classés)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# -----------------------------
# Utilitaires numériques
# -----------------------------

def one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=int).ravel()
    Y = np.zeros((y.shape[0], num_classes), dtype=float)
    Y[np.arange(y.shape[0]), y] = 1.0
    return Y


def softmax(Z: np.ndarray) -> np.ndarray:
    # Softmax stable numériquement
    Z = Z - np.max(Z, axis=1, keepdims=True)
    expZ = np.exp(Z)
    return expZ / np.sum(expZ, axis=1, keepdims=True)


def add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])


@dataclass
class StandardScaler:
    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0.0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler must be fitted before transform().")
        X = np.asarray(X, dtype=float)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def stratified_split(y: np.ndarray, train_size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split stratifié retournant (train_idx, test_idx) avec train_size EXACT.

    Objectif : reproduire un split stable (utile pour comparaison et rapport).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=int).ravel()
    n = y.shape[0]
    if not (0 < train_size < n):
        raise ValueError("train_size must be in (0, n).")

    classes, counts = np.unique(y, return_counts=True)

    desired_train = counts * (train_size / n)
    per_class_train = np.floor(desired_train).astype(int)

    diff = train_size - per_class_train.sum()
    if diff != 0:
        frac = desired_train - np.floor(desired_train)
        order = np.argsort(frac)[::-1]
        step = 1 if diff > 0 else -1
        remaining = abs(diff)
        k = 0
        while remaining > 0:
            c_idx = order[k % len(order)]
            new_val = per_class_train[c_idx] + step
            if 0 < new_val < counts[c_idx]:
                per_class_train[c_idx] = new_val
                remaining -= 1
            k += 1

    train_idx: List[int] = []
    test_idx: List[int] = []
    for c, n_train_c in zip(classes, per_class_train):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:n_train_c].tolist())
        test_idx.extend(idx[n_train_c:].tolist())

    train_idx = np.array(train_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def classification_metrics_from_cm(cm: np.ndarray) -> Dict[str, float]:
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    acc = tp.sum() / cm.sum() if cm.sum() else 0.0
    return {
        "accuracy": float(acc),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
    }


def log_loss(y_true: np.ndarray, proba: np.ndarray, eps: float = 1e-15) -> float:
    y_true = np.asarray(y_true, dtype=int).ravel()
    proba = np.asarray(proba, dtype=float)
    proba = np.clip(proba, eps, 1.0 - eps)
    return float(-np.mean(np.log(proba[np.arange(y_true.shape[0]), y_true])))


# -----------------------------
# Modèle from-scratch (GD)
# -----------------------------

class SoftmaxRegressionGD:
    """
    Régression logistique multi-classes (softmax) optimisée par descente de gradient (batch).

    Paramètres
    ----------
    alpha : float
        Taux d'apprentissage.
    max_epochs : int
        Nombre maximal d'itérations (epochs).
    eps : float
        Critère d'arrêt : ||grad||_F <= eps.
    seed : int
        Seed pour l'init aléatoire (si init='random').
    init : str
        'zeros' ou 'random'.
    """

    def __init__(
        self,
        alpha: float = 0.3,
        max_epochs: int = 500,
        eps: float = 1e-5,
        seed: int = 42,
        init: str = "zeros",
    ) -> None:
        self.alpha = float(alpha)
        self.max_epochs = int(max_epochs)
        self.eps = float(eps)
        self.seed = int(seed)
        if init not in {"zeros", "random"}:
            raise ValueError("init must be 'zeros' or 'random'.")
        self.init = init

        self.theta_: Optional[np.ndarray] = None  # (C, D)
        self.history_: Dict[str, List[float]] = {"loss": [], "grad_norm": []}

    def fit(self, X: np.ndarray, y: np.ndarray, num_classes: Optional[int] = None) -> "SoftmaxRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int).ravel()
        n, d = X.shape

        if num_classes is None:
            num_classes = int(y.max() + 1)
        C = int(num_classes)

        Y = one_hot(y, C)

        if self.init == "zeros":
            theta = np.zeros((C, d), dtype=float)
        else:
            rng = np.random.default_rng(self.seed)
            theta = rng.normal(loc=0.0, scale=0.01, size=(C, d))

        for _ in range(self.max_epochs):
            scores = X @ theta.T          # (n, C)
            P = softmax(scores)           # (n, C)

            # Entropie croisée moyenne
            loss = -np.mean(np.sum(Y * np.log(P + 1e-15), axis=1))

            # Gradient : (C, D)
            grad = ((P - Y).T @ X) / n
            grad_norm = float(np.linalg.norm(grad))

            self.history_["loss"].append(float(loss))
            self.history_["grad_norm"].append(grad_norm)

            if grad_norm <= self.eps:
                break

            theta = theta - self.alpha * grad

        self.theta_ = theta
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.theta_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X, dtype=float)
        return softmax(X @ self.theta_.T)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# -----------------------------
# Plots (optionnels)
# -----------------------------

def plot_learning_curve(loss: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.plot(np.arange(len(loss)), loss)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_confusion(cm: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_weight_maps(W: np.ndarray, out_path: Path, title: str) -> None:
    """
    W attendu shape (10, 64) pour digits.
    """
    plt.figure(figsize=(10, 4))
    for c in range(10):
        ax = plt.subplot(2, 5, c + 1)
        ax.imshow(W[c].reshape(8, 8), interpolation="nearest", cmap="gray")
        ax.set_title(str(c))
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_misclassified(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path, title: str) -> None:
    wrong = np.where(y_true != y_pred)[0]
    n_show = min(16, wrong.shape[0])
    plt.figure(figsize=(8, 8))
    for i in range(n_show):
        idx = wrong[i]
        ax = plt.subplot(4, 4, i + 1)
        ax.imshow(images[idx].reshape(8, 8), interpolation="nearest", cmap="gray")
        ax.set_title(f"t={y_true[idx]}, p={y_pred[idx]}")
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# -----------------------------
# API utilisée par la partie 3
# -----------------------------

def train_eval_scratch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    images_test: Optional[np.ndarray] = None,
    *,
    alpha: float = 0.3,
    max_epochs: int = 500,
    eps: float = 1e-5,
    seed: int = 42,
    out_dir: Optional[Path] = None,
    prefix: str = "scratch",
    no_plots: bool = False,
) -> Dict[str, Any]:
    model = SoftmaxRegressionGD(alpha=alpha, max_epochs=max_epochs, eps=eps, seed=seed, init="zeros")
    model.fit(X_train, y_train, num_classes=10)

    proba_test = model.predict_proba(X_test)
    y_pred = np.argmax(proba_test, axis=1)

    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    metrics = classification_metrics_from_cm(cm)
    metrics["log_loss"] = log_loss(y_test, proba_test)
    metrics["epochs_ran"] = len(model.history_["loss"])

    out: Dict[str, Any] = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }

    if out_dir is not None and not no_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_learning_curve(np.array(model.history_["loss"]), out_dir / f"{prefix}_loss.png", "From-scratch loss")
        plot_confusion(cm, out_dir / f"{prefix}_confusion.png", "From-scratch confusion matrix")

        # W = theta sans biais (1ère colonne)
        W = model.theta_[:, 1:]  # (10, 64)
        plot_weight_maps(W, out_dir / f"{prefix}_weights.png", "From-scratch learned weights")

        if images_test is not None:
            plot_misclassified(images_test, y_test, y_pred, out_dir / f"{prefix}_misclassified.png", "From-scratch misclassified")

    return out


# -----------------------------
# CLI (Partie 1 seule)
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=1437)  # ~80% (comme ton rapport)
    parser.add_argument("--out", type=str, default="outputs_part1")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    outputs = Path(args.out)
    outputs.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X = digits.data.astype(float)          # (1797, 64)
    y = digits.target.astype(int)          # (1797,)
    images = digits.images.reshape(-1, 64) # pour affichage

    train_idx, test_idx = stratified_split(y, train_size=args.train_size, seed=args.seed)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    images_test = images[test_idx]

    # Standardisation (fit sur train)
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    # Ajout biais
    X_train = add_bias(X_train_std)
    X_test = add_bias(X_test_std)

    res = train_eval_scratch(
        X_train, y_train, X_test, y_test, images_test,
        alpha=args.alpha, max_epochs=args.max_epochs, eps=args.eps, seed=args.seed,
        out_dir=outputs, prefix="scratch", no_plots=args.no_plots,
    )

    payload = {
        "split": {"train_size": int(train_idx.shape[0]), "test_size": int(test_idx.shape[0])},
        "preprocessing": "standardization (fit on train) + bias column",
        "scratch": res,
    }
    (outputs / "results_part1_scratch.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Split:", payload["split"])
    print("Scratch metrics:", payload["scratch"]["metrics"])
    print(f"Outputs: {outputs.resolve()}")


if __name__ == "__main__":
    main()
