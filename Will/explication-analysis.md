### Script 3 : Visualisation et Analyse des Résultats (`analysis.py`)

Ce script ne se contente pas de donner des chiffres ; il "ouvre" ton modèle pour voir ce qu'il a appris et pourquoi il fait des erreurs.

#### 1. Explications du script (Les 3 points clés pour toi)

1.
**Interprétation des coefficients (Poids appris) :** Ton rapport mentionne une matrice  de dimension . Ce script va extraire ces coefficients et les réorganiser en images de  pixels.


* **Pourquoi ?** Cela permet de voir la "forme idéale" que ton modèle a construite pour chaque chiffre. Un coefficient élevé (rouge) indique qu'un pixel est déterminant pour reconnaître ce chiffre.


2. **Analyse des erreurs (Matrice de Confusion) :** Au lieu de simplement dire "j'ai 95% de réussite", on utilise une matrice de confusion. Elle montre exactement quels chiffres sont confondus (par exemple, si le modèle prend souvent un `9` pour un `3`). C'est l'analyse de recul demandée dans les objectifs.


3. **Visualisation des échecs :** Le script affiche les images que le modèle "From Scratch" n'a pas réussi à classer. C'est essentiel pour "proposer des explications des erreurs" comme indiqué dans les consignes du projet.

---

### Synthèse du Projet

Avec ces trois scripts, tu couvres l'intégralité du barème :

1.
**Théorie et Implémentation :** Respect du rapport avec Softmax, One-Hot et Gradient Matriciel.


2.
**Outils du cours :** Utilisation obligatoire de `my_descent_gradient.py`.


3.
**Comparaison :** Résultats mis en parallèle avec Scikit-Learn (en ayant corrigé le Warning).


4.
**Analyse Profonde :** Interprétation visuelle des poids et étude des erreurs de classification.



**Une précision importante :** Dans ton fichier `main.py`, n'oublie pas de standardiser les données. Sans cela, les exponentielles du Softmax pourraient produire des valeurs infinies (`NaN`), ce qui ferait échouer ta descente de gradient.