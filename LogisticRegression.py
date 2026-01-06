#!/usr/bin/env python3
"""
Partie 2 — Régression logistique multi-classes via scikit-learn (baseline)
Dataset : sklearn.datasets.load_digits

Exécution :
    python LogisticRegression.py
    python LogisticRegression.py --seed 42 --train_size 1437

Sorties :
    - results_part2_sklearn.json
    - figures PNG (matrice de confusion, poids 8x8, exemples mal classés)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Pré-traitement / split
# (dupliqué ici pour que la partie 2 soit autonome)
# -----------------------------

def add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    return np.hstack([np.ones((X.shape[0], 1), dtype=float), X])


class StandardScaler:
    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

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
# Plots
# -----------------------------

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

def train_eval_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    images_test: Optional[np.ndarray] = None,
    *,
    seed: int = 42,
    max_iter: int = 2000,
    out_dir: Optional[Path] = None,
    prefix: str = "sklearn",
    no_plots: bool = False,
) -> Dict[str, Any]:
    """
    On garde fit_intercept=False car on ajoute un biais manuellement (colonne de 1),
    pour matcher la convention de la partie 1.
    """
    try:
        clf = LogisticRegression(
            penalty="none",
            solver="lbfgs",
            fit_intercept=False,
            max_iter=max_iter,
            random_state=seed,
        )
        clf.fit(X_train, y_train)
    except Exception:
        # Fallback : L2 très faible (C très grand) si penalty='none' non supporté
        clf = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            fit_intercept=False,
            max_iter=max_iter,
            random_state=seed,
        )
        clf.fit(X_train, y_train)

    proba_test = clf.predict_proba(X_test)
    y_pred = np.argmax(proba_test, axis=1)

    cm = confusion_matrix(y_test, y_pred, num_classes=10)
    metrics = classification_metrics_from_cm(cm)
    metrics["log_loss"] = log_loss(y_test, proba_test)

    out: Dict[str, Any] = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
    }

    if out_dir is not None and not no_plots:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_confusion(cm, out_dir / f"{prefix}_confusion.png", "Scikit-Learn confusion matrix")

        # coef_ inclut la feature biais (première colonne) car fit_intercept=False + bias ajouté dans X
        W = clf.coef_
        if W.shape[1] == 65:
            W = W[:, 1:]
        plot_weight_maps(W, out_dir / f"{prefix}_weights.png", "Scikit-Learn learned weights")

        if images_test is not None:
            plot_misclassified(images_test, y_test, y_pred, out_dir / f"{prefix}_misclassified.png", "Scikit-Learn misclassified")

    return out


# -----------------------------
# CLI (Partie 2 seule)
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=1437)
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--out", type=str, default="outputs_part2")
    parser.add_argument("--no_plots", action="store_true")
    args = parser.parse_args()

    outputs = Path(args.out)
    outputs.mkdir(parents=True, exist_ok=True)

    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(int)
    images = digits.images.reshape(-1, 64)

    train_idx, test_idx = stratified_split(y, train_size=args.train_size, seed=args.seed)

    X_train_raw, y_train = X[train_idx], y[train_idx]
    X_test_raw, y_test = X[test_idx], y[test_idx]
    images_test = images[test_idx]

    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_raw)
    X_test_std = scaler.transform(X_test_raw)

    X_train = add_bias(X_train_std)
    X_test = add_bias(X_test_std)

    res = train_eval_sklearn(
        X_train, y_train, X_test, y_test, images_test,
        seed=args.seed, max_iter=args.max_iter,
        out_dir=outputs, prefix="sklearn", no_plots=args.no_plots,
    )

    payload = {
        "split": {"train_size": int(train_idx.shape[0]), "test_size": int(test_idx.shape[0])},
        "preprocessing": "standardization (fit on train) + bias column",
        "sklearn": res,
    }
    (outputs / "results_part2_sklearn.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Split:", payload["split"])
    print("Sklearn metrics:", payload["sklearn"]["metrics"])
    print(f"Outputs: {outputs.resolve()}")


if __name__ == "__main__":
    main()
