Voici une proposition structurée pour la section **« Analyse et Interprétation »** de ton rapport. Cette partie est essentielle pour démontrer au professeur que tu as compris non seulement le code, mais aussi la logique mathématique et les limites du modèle.

---

## 3 Analyse et Interprétation des résultats

### 3.1 Influence des paramètres et convergence

L'implémentation « From Scratch » repose sur une descente de gradient classique utilisant l'outil `GradientDescent`. Plusieurs facteurs ont influencé la qualité de l'apprentissage :

* 
**Standardisation :** Comme prévu dans la méthodologie, la transformation  a été indispensable pour éviter le phénomène d'**overflow** dans la fonction Softmax. Sans elle, les valeurs exponentielles calculées par  dépassent les capacités de stockage numérique de Python.


* 
**Nombre d'époques () :** Nous avons observé qu'un minimum de 500 à 1000 itérations est nécessaire pour que la norme du gradient  devienne négligeable.


* 
**Taux d'apprentissage () :** Un pas trop grand provoque des oscillations de la fonction d'erreur , tandis qu'un pas trop petit ralentit excessivement la convergence vers le minimum global garanti par la convexité de la fonction.



### 3.2 Interprétation des coefficients appris ()

La matrice de paramètres  apprise, de dimension  (soit ), contient l'essence de la reconnaissance visuelle du modèle. En isolant chaque ligne de cette matrice (en ignorant le biais ) et en la replaçant dans un format , nous obtenons une représentation visuelle des poids:

* 
**Poids positifs (zones rouges) :** Indiquent les pixels qui, s'ils sont allumés, augmentent la probabilité que l'image appartienne à cette classe.


* **Poids négatifs (zones bleues) :** Indiquent les pixels qui discréditent l'appartenance à la classe.
* 
**Observation :** Pour le chiffre "0", on observe souvent un cercle de poids positifs entourant une zone neutre, ce qui correspond fidèlement à la topologie du chiffre manuscrit.



### 3.3 Comparaison avec Scikit-Learn

Nous avons comparé notre modèle avec l'implémentation `LogisticRegression` de Scikit-Learn.

| Métrique | Implémentation "From Scratch" | Scikit-Learn |
| --- | --- | --- |
| **Précision (Accuracy)** | ~94-96% | ~96-98% |
| **Entropie Croisée (Log Loss)** | Faible | Très faible |

**Analyse de l'écart :** Bien que notre modèle soit très performant, Scikit-Learn utilise des algorithmes d'optimisation plus avancés que la descente de gradient simple, ce qui lui permet d'atteindre le minimum de la fonction d'erreur plus rapidement et avec une précision légèrement supérieure.

### 3.4 Analyse des erreurs et limites

L'étude de la **matrice de confusion** révèle que les erreurs ne sont pas aléatoires:

* **Confusions fréquentes :** Le modèle confond parfois les chiffres  et  ou  et . Cela s'explique par la faible résolution des images ( pixels), où la perte d'un seul pixel clé peut transformer radicalement la signature numérique du chiffre.


* **Limites du modèle :** La régression logistique est un classifieur linéaire. Elle peine à capturer des relations spatiales complexes ou des déformations importantes (rotations, inclinaisons) contrairement à des modèles non-linéaires ou des réseaux de neurones profonds.



---

## 4 Conclusion

Ce projet nous a permis de valider l'efficacité de la régression logistique multi-classe pour la reconnaissance de formes simples. L'implémentation « From Scratch » de l'opérateur Softmax et du gradient matriciel  démontre une maîtrise de la théorie mathématique sous-jacente. Malgré des performances légèrement en deçà des outils optimisés du marché, la compréhension interne des mécanismes d'apprentissage reste un atout majeur pour le diagnostic et l'amélioration future des modèles de Data Science.

---

### Ouverture Critique

1. 
**Sur la méthodologie :** Le choix d'un découpage 80/20  est standard, mais l'utilisation d'une **validation croisée** (Cross-Validation) aurait été plus rigoureuse pour s'assurer que les performances ne dépendent pas d'un tirage chanceux des données de test.


2. **Angle complémentaire :** Il serait intéressant de tester l'impact de la **régularisation L2 (Ridge)**. En ajoutant un terme de pénalité sur la norme de , on pourrait réduire le sur-apprentissage (overfitting) et potentiellement lisser les coefficients appris pour les rendre plus lisibles visuellement.