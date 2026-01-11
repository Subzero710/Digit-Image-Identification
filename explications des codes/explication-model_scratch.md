
---

### 1. Explications du script (Les 3 points clés)

Voici les trois piliers du code pour répondre aux exigences du projet :

1. **La structure "Conteneur" :** Au lieu d'avoir des variables qui traînent partout, on regroupe tout dans une structure `ModelScratch`. Elle va stocker la matrice  (poids) de dimension . Cela permet de séparer proprement tes calculs "maison" de ceux de Scikit-Learn plus tard.


2. **L'implémentation de la "Vérité Mathématique" :** Le script suit scrupuleusement les formules de ton rapport :
* La fonction **Softmax** pour transformer les scores en probabilités .


* Le **Gradient Matriciel** : . C'est ici que réside la performance "from scratch".




3. **Le Pont avec l'outil du Professeur :** J'ai intégré l'import de `my_descent_gradient.py`. Dans la phase d'entraînement, le script instancie l'objet `GradientDescent` avec ta fonction de gradient personnalisée. C'est ce qui fait le lien entre la théorie mathématique et l'exécution informatique.

---

### Ouverture Critique

* **Pertinence de la structure :** Bien que tu préfères les fonctions simples des TDs, l'utilisation d'une classe ici permet de "sauvegarder" le `self.theta` après l'entraînement. Sans cela, tu perdrais tes résultats dès que la fonction se termine. C'est essentiel pour la suite (comparaison et analyse).
*
**Optimisation :** Pour l'analyse demandée sur "l'influence des paramètres", cette structure te permettra très facilement d'extraire `self.theta` et de l'afficher sous forme d'image  pour voir ce que le modèle "regarde".