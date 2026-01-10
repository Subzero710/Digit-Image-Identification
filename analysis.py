# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def plot_coefficients(theta):
    """
    Visualise les poids appris pour chaque chiffre (0-9).
    On ignore la première colonne (le biais).
    """
    # On retire la colonne du biais pour avoir 64 pixels
    weights = theta[:, 1:]

    plt.figure(figsize=(12, 5))
    plt.suptitle("Visualisation des coefficients appris (Poids par pixel)")

    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # On replace les 64 coefficients en forme 8x8
        weight_image = weights[i].reshape(8, 8)
        plt.imshow(weight_image, cmap='RdBu_r', interpolation='nearest')
        plt.title(f"Chiffre {i}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, title="Matrice de Confusion"):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(title)


def analyze_errors(model, X_test, y_test, X_test_raw):
    """Affiche quelques exemples d'images mal classées."""
    y_pred = model.predict(X_test)
    errors = np.where(y_pred != y_test)[0]

    if len(errors) > 0:
        plt.figure(figsize=(10, 4))
        plt.suptitle("Exemples d'erreurs du modèle 'From Scratch'")
        # On affiche les 5 premières erreurs
        for i, idx in enumerate(errors[:5]):
            plt.subplot(1, 5, i + 1)
            plt.imshow(X_test_raw[idx].reshape(8, 8), cmap='gray')
            plt.title(f"Vrai:{y_test[idx]}\nPrédit:{y_pred[idx]}")
            plt.axis('off')
        plt.show()
    else:
        print("Aucune erreur détectée sur le jeu de test !")

