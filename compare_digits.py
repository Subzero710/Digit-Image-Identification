#!/usr/bin/env python3
"""
Partie 3 — Comparaison des résultats (scratch vs scikit-learn)
- même split
- même pré-traitement (standardisation + biais)
- export d'un JSON comparatif + (optionnel) figures PNG

Exécution :
    python compare_digits.py
    python compare_digits.py --alpha 0.3 --max_epochs 500 --eps 1e-5 --seed 42 --train_size 1437
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
from sklearn.datasets import load_digits

# Import des 2 parties précédentes (elles n'exécutent rien au import)
from gradient_descent import train_eval_scratch
from LogisticRegression import train_eval_sklearn


# -----------------------------
# Pré-traitement / split (pour garantir exactement le même)
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


# -----------------------------
# CLI (Partie 3)
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_size", type=int, default=1437)
    parser.add_argument("--max_iter", type=int, default=2000)  # sklearn
    parser.add_argument("--out", type=str, default="outputs_part3")
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

    scratch = train_eval_scratch(
        X_train, y_train, X_test, y_test, images_test,
        alpha=args.alpha, max_epochs=args.max_epochs, eps=args.eps, seed=args.seed,
        out_dir=outputs, prefix="scratch", no_plots=args.no_plots,
    )
    skl = train_eval_sklearn(
        X_train, y_train, X_test, y_test, images_test,
        seed=args.seed, max_iter=args.max_iter,
        out_dir=outputs, prefix="sklearn", no_plots=args.no_plots,
    )

    payload: Dict[str, Any] = {
        "split": {"train_size": int(train_idx.shape[0]), "test_size": int(test_idx.shape[0])},
        "preprocessing": "standardization (fit on train) + bias column",
        "scratch": scratch,
        "sklearn": skl,
        "delta_metrics": {
            k: float(scratch["metrics"].get(k, 0.0) - skl["metrics"].get(k, 0.0))
            for k in set(scratch["metrics"].keys()) | set(skl["metrics"].keys())
        },
        "hyperparams": {
            "scratch": {"alpha": args.alpha, "max_epochs": args.max_epochs, "eps": args.eps, "seed": args.seed},
            "sklearn": {"max_iter": args.max_iter, "seed": args.seed},
        },
    }

    (outputs / "results_part3_compare.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Split:", payload["split"])
    print("Scratch metrics:", payload["scratch"]["metrics"])
    print("Sklearn metrics:", payload["sklearn"]["metrics"])
    print("Delta (scratch - sklearn):", payload["delta_metrics"])
    print(f"Outputs: {outputs.resolve()}")


if __name__ == "__main__":
    main()
