# -*- coding: utf-8 -*-
"""
Script principal d'exécution du projet Digits.
Effectue le chargement, la standardisation et la comparaison.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss


# Import de notre modèle "Maison"
from model_scratch import LogitScratch

from analysis import plot_coefficients, plot_confusion, analyze_errors

# 1. Chargement des données
digits = load_digits()
X, y = digits.data, digits.target
N = len(y)  # 1797 images [cite: 42]

# 2. Découpage Train/Test (80% / 20%)
# On fixe random_state pour avoir des résultats reproductibles
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Standardisation
# Indispensable pour la convergence de la descente de gradient
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PHASE 1 : Entraînement "From Scratch" ---
print("Entraînement du modèle 'From Scratch'...")
# Paramètres : learning_rate=0.1, iterations=500 (ajustables)
model_home = LogitScratch(lr=0.1, iterations=500)
model_home.fit(X_train_scaled, y_train)

# --- PHASE 2 : Entraînement Scikit-Learn  ---
print("Entraînement du modèle 'Scikit-Learn'...")
model_sk = LogisticRegression(solver='lbfgs', max_iter=500)
model_sk.fit(X_train_scaled, y_train)


# --- PHASE 3 : Comparaison des résultats ---

def evaluate(model, X, y, name):
    # Prédictions
    y_pred = model.predict(X)

    # Calcul des probabilités pour le Log Loss (Entropie croisée) [cite: 63, 70]
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X)
    else:
        # Pour notre modèle scratch, on peut appeler la méthode interne
        from model_scratch import LogitScratch
        # On recrée les probas si nécessaire ou on ajoute predict_proba à LogitScratch
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        probas = model._softmax(X_bias @ model.theta.T)

    acc = accuracy_score(y, y_pred)
    loss = log_loss(y, probas)

    print(f"\nRésultats {name}:")
    print(f" - Accuracy : {acc:.4f}")
    print(f" - Log Loss : {loss:.4f}")


evaluate(model_home, X_test_scaled, y_test, "MAISON (From Scratch)")
evaluate(model_sk, X_test_scaled, y_test, "SCIKIT-LEARN")

# Ana
plot_coefficients(model_home.theta)
plot_confusion(y_test, model_home.predict(X_test_scaled), "Confusion: From Scratch")
analyze_errors(model_home, X_test_scaled, y_test, X_test)