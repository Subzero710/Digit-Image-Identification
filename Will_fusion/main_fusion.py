# -*- coding: utf-8 -*-
"""
Inclut : Modèle Maison (via GradientDescent), Scikit-Learn et Bonus Keras.

gardé pour comparer au main_fusion_3
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix

# Imports locaux des modules que tu as fournis
from model_scratch import LogitScratch
from analysis import plot_coefficients, plot_confusion
# On importe la fonction de visualisation d'exemples du script d'analyse
from analyze_digits_results import show_examples

# 1. CHARGEMENT ET PRÉPARATION (Respect du Rapport )
digits = load_digits()
X, y = digits.data, digits.target
images = digits.images

# Découpage 80% / 20% avec stratification pour garder l'équilibre des classes
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.2, random_state=42, stratify=y
)

# Standardisation [cite: 45]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PHASE 1 : MODÈLE FROM SCRATCH ---
print("\n--- PHASE 1 : Entraînement du modèle 'Maison' (via GradientDescent) ---")
# On utilise les paramètres optimisés pour la convergence
model_home = LogitScratch(lr=0.1, iterations=1000)
model_home.fit(X_train_scaled, y_train)

# --- PHASE 2 : SCIKIT-LEARN ---
print("--- PHASE 2 : Entraînement du modèle 'Scikit-Learn' ---")
# Correction du warning multi_class pour sklearn 1.5+
model_sk = LogisticRegression(solver='lbfgs', max_iter=1000)
model_sk.fit(X_train_scaled, y_train)

# --- PHASE 3 : BONUS (Réseau de Neurones) ---
print("--- PHASE 3 : Entraînement du modèle Bonus (TensorFlow/Keras) ---")
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Normalisation simple pour le réseau de neurones (0 à 1)
X_train_nn = X_train / 16.0
X_test_nn = X_test / 16.0

model_nn = Sequential([
    Dense(16, activation="relu", input_shape=(64,)),
    Dense(10, activation="softmax")
])
model_nn.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_nn.fit(X_train_nn, y_train, epochs=20, batch_size=32, verbose=0)

# --- ÉVALUATION ET COMPARAISON ---

def full_report(model, X, y, name, is_keras=False):
    if is_keras:
        probas = model.predict(X, verbose=0)
        y_pred = np.argmax(probas, axis=1)
    else:
        y_pred = model.predict(X)
        # Gestion des probabilités pour le log_loss
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
        else:
            # Re-calcul des probabilités pour le modèle scratch
            X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
            probas = model._softmax(X_bias @ model.theta.T)

    print(f"\n==================== {name.upper()} ====================")
    print(f"Accuracy : {accuracy_score(y, y_pred):.4f}")
    print(f"Log Loss : {log_loss(y, probas):.4f}")
    print("\nClassification Report :")
    print(classification_report(y, y_pred, digits=4))
    return y_pred

# Lancement des évaluations
y_pred_home = full_report(model_home, X_test_scaled, y_test, "Maison (Scratch)")
y_pred_sk = full_report(model_sk, X_test_scaled, y_test, "Scikit-Learn")
y_pred_nn = full_report(model_nn, X_test_nn, y_test, "Bonus (Keras)", is_keras=True)

# --- VISUALISATION DES RÉSULTATS ---

# 1. Matrice de confusion pour le modèle Maison
plot_confusion(y_test, y_pred_home, "Matrice de Confusion : Modèle Maison")

# 2. Visualisation des coefficients (Interprétation des poids)
plot_coefficients(model_home.theta)

# 3. Analyse visuelle des erreurs du modèle Maison
err_idx = np.where(y_pred_home != y_test)[0]
ok_idx = np.where(y_pred_home == y_test)[0]

show_examples("Erreurs du modèle Maison", img_test, y_test, y_pred_home, err_idx)
show_examples("Succès du modèle Maison", img_test, y_test, y_pred_home, ok_idx)

plt.show()