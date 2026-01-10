# -*- coding: utf-8 -*-
"""
gardé pour comparer les commentaires au main_fusion_3


Inclut : Modèle Maison, Comparaison Scikit-Learn, Étude de Régularisation et Bonus Keras.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

# Imports des modules locaux (Assurez-vous qu'ils sont dans le même dossier)
from model_scratch import LogitScratch
from analysis import plot_coefficients, plot_confusion
from analyze_digits_results import show_examples # Note: Commentez l'import de softmax_regression dans ce fichier

# Import pour la partie bonus (phase 3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# [cite_start]1. PRÉPARATION DES DONNÉES (Respect strict du rapport [cite: 42, 43, 45])
digits = load_digits()
X, y = digits.data, digits.target
images = digits.images

# [cite_start]Découpage 80% / 20% (1437 images d'entraînement [cite: 43])
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.2, random_state=42, stratify=y
)

# [cite_start]Standardisation x_std = (x - mean) / sigma [cite: 45]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# [cite_start]--- PHASE 1 : MODÈLE FROM SCRATCH (Obligatoire [cite: 12]) ---
print("\n--- PHASE 1 : Modèle 'Maison' (via GradientDescent du cours) ---")
model_home = LogitScratch(lr=0.1, iterations=1000)
model_home.fit(X_train_scaled, y_train)

# [cite_start]--- PHASE 2 : SCIKIT-LEARN & ÉTUDE DE LA RÉGULARISATION (Bonus [cite: 13, 15]) ---
print("\n--- PHASE 2 : Comparaison Sklearn & Étude de la régularisation L2 ---")

# [cite_start]2.1 Sklearn Standard (Comparaison de base [cite: 13])
model_sk_std = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0)
model_sk_std.fit(X_train_scaled, y_train)

# [cite_start]2.2 Sklearn Régularisé (Étude facultative de la régularisation [cite: 15])
# C=0.01 impose une forte pénalité L2 pour stabiliser les poids
model_sk_reg = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01)
model_sk_reg.fit(X_train_scaled, y_train)

# [cite_start]--- PHASE 3 : RÉSEAU DE NEURONES (Bonus Facultatif [cite: 15]) ---
print("\n--- PHASE 3 : Méthode alternative (Réseau de Neurones Keras) ---")

# Normalisation 0-1 pour le Deep Learning
X_train_nn, X_test_nn = X_train / 16.0, X_test / 16.0

model_nn = Sequential([
    Dense(16, activation="relu", input_shape=(64,)),
    Dense(10, activation="softmax")
])
model_nn.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_nn.fit(X_train_nn, y_train, epochs=20, batch_size=32, verbose=0)

# --- FONCTION D'ÉVALUATION ET RAPPORTS ---

def evaluate_all(model, X, y, name, is_keras=False):
    if is_keras:
        probas = model.predict(X, verbose=0)
        y_pred = np.argmax(probas, axis=1)
    else:
        y_pred = model.predict(X)
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
        else:
            # [cite_start]Calcul probas pour modèle maison (Softmax du rapport [cite: 56])
            X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
            probas = model._softmax(X_bias @ model.theta.T)

    print(f"\n>> RÉSULTATS : {name}")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Log Loss : {log_loss(y, probas):.4f}")
    print(classification_report(y, y_pred, digits=4))
    return y_pred

# Affichage des rapports dans la console
y_pred_home = evaluate_all(model_home, X_test_scaled, y_test, "MAISON (SCRATCH)")
y_pred_sk = evaluate_all(model_sk_std, X_test_scaled, y_test, "SKLEARN STANDARD")
y_pred_reg = evaluate_all(model_sk_reg, X_test_scaled, y_test, "SKLEARN RÉGULARISÉ (BONUS)")
y_pred_nn = evaluate_all(model_nn, X_test_nn, y_test, "KERAS (BONUS)", is_keras=True)

# [cite_start]--- VISUALISATION (Analyse des résultats [cite: 14]) ---

# Matrice de confusion du modèle Maison
plot_confusion(y_test, y_pred_home, "Matrice de Confusion : Modèle Maison")

# [cite_start]Interprétation des poids appris (Coefficients Theta [cite: 51])
plot_coefficients(model_home.theta)

# Visualisation des erreurs spécifiques
err_idx = np.where(y_pred_home != y_test)[0]
show_examples("Erreurs du modèle Maison", img_test, y_test, y_pred_home, err_idx)

plt.show()