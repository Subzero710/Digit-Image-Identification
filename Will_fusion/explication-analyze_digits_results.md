### 1. Analyse du script `analyze_digits_results.py`

**module de services** que le `main.py` appelle pour générer des statistiques professionnelles.

* **Fonction `per_class_acc**` : Elle extrait la diagonale de la matrice de confusion pour calculer le taux de réussite spécifique à chaque chiffre (de 0 à 9). Cela permet de voir si le modèle est plus faible sur certains chiffres complexes comme le "8" ou le "9".
* **Fonction `report**` : C'est le centre de diagnostic textuel. Elle calcule l'**Accuracy**, le **Log Loss** (l'entropie croisée définie dans ton rapport) et génère le `classification_report`. Ce rapport inclut la Précision et le Rappel, ce qui est indispensable pour l'analyse de recul demandée par le projet.
* **Fonction `show_examples**` : Elle crée une grille d'images (12 par défaut) pour comparer visuellement la prédiction (`g` pour guess) et la réalité (`r` pour real). C'est l'outil principal pour ta partie "explication des erreurs".
