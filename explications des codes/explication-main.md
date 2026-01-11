## 2. Analyse Globale : `main.py`

Ce script est le chef d'orchestre de ton projet. Il gère l'intégralité du pipeline, du chargement des pixels à la visualisation finale.

### A. Préparation des Données

* **Chargement et Découpage** : Le script charge les 1797 images et respecte ton rapport en isolant 20% des données pour le test (le modèle ne voit jamais ces images durant l'entraînement).
* **Standardisation (`StandardScaler`)** : Il transforme les pixels pour qu'ils aient une moyenne de 0 et un écart-type de 1. Cette étape est cruciale pour ton modèle "Maison" car elle empêche les exponentielles du Softmax d'exploser numériquement (overflow).

### B. Entraînement Multi-Modèles

Le script lance quatre entraînements distincts pour permettre une comparaison exhaustive :

1. **Le modèle "Maison" (`ModelScratch`)** : Il utilise ta classe personnalisée qui fait appel à l'outil `GradientDescent` du professeur. C'est le cœur de ton projet "From Scratch".
2. **Sklearn Standard** : Il sert de base de comparaison industrielle.
3. **Sklearn Régularisé (Bonus L2)** : Il utilise un paramètre  pour imposer une forte pénalité sur les poids, répondant ainsi au point facultatif sur l'étude de la régularisation.
4. **Keras (Bonus Deep Learning)** : Il entraîne un réseau de neurones avec une couche cachée de 16 neurones et une activation ReLU pour tester une méthode non-linéaire.

### C. Boucle d'Analyse Unifiée (`analyze_model`)

C'est la partie la plus importante du nouveau `main.py`. J'ai créé une fonction qui traite chaque modèle de la même manière :

* **Unification des prédictions** : Elle gère la différence de format entre Keras (qui renvoie des probabilités) et les modèles linéaires.
* **Génération des rapports** : Pour **chaque modèle**, elle appelle les fonctions de `analyze_digits_results.py` pour imprimer les scores dans la console.
* **Visualisation des erreurs** : Elle génère automatiquement une matrice de confusion et une grille d'erreurs visuelles pour chaque modèle.
