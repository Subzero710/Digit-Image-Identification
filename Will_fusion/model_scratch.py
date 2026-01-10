# -*- coding: utf-8 -*-
import numpy as np
from my_descent_gradient import GradientDescent


class LogitScratch:
    def __init__(self, lr, iterations):
        """Initialise les hyperparamètres."""
        self.lr = lr
        self.iterations = iterations
        self.theta = None  # Sera une matrice (C, p+1)

    def _softmax(self, Z):
        """Formule g_Theta(x) du rapport."""
        # Stabilité numérique (évite overflow)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def _one_hot(self, y, C):
        """Encodage One-Hot de la sortie réelle."""
        return np.eye(C)[y.astype(int)]

    def fit(self, X, y):
        """Entraîne le modèle en utilisant l'outil du professeur."""
        # 1. Préparation selon le rapport
        N = X.shape[0]
        C = 10  # 10 classes (chiffres 0 à 9)

        # Ajout du '1' au début pour le biais
        X_bias = np.hstack((np.ones((N, 1)), X))
        Y_encoded = self._one_hot(y, C)

        # Initialisation de Theta à zéro (C, p+1)
        initial_theta = np.zeros((C, X_bias.shape[1]))

        # 2. Définition de la fonction Gradient pour l'outil du prof
        # Cette fonction doit prendre UNIQUEMENT theta en argument
        def grad_func(current_theta):
            # Calcul de Pi (probabilités prédites)
            Pi = self._softmax(X_bias @ current_theta.T)
            # Formule exacte du rapport : (1/N) * (Pi - Y)^T * X
            return (1 / N) * ((Pi - Y_encoded).T @ X_bias)

        optimizer = GradientDescent(gradient=grad_func,
                                    learning_rate=self.lr,
                                    max_iterations=self.iterations)

        # Lancement de la descente
        self.theta = optimizer.descent(initial_theta)

    def predict(self, X):
        """Règle de décision : argmax[cite: 60]."""
        N = X.shape[0]
        X_bias = np.hstack((np.ones((N, 1)), X))
        # f(x) = argmax( Theta * x )
        probas = self._softmax(X_bias @ self.theta.T)
        return np.argmax(probas, axis=1)