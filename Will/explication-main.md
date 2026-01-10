
### Script 2 : Exécution, Comparaison et Performance (`main.py`)

Ce script est le chef d'orchestre. Il utilise le "moteur" que nous avons créé au message précédent pour répondre aux objectifs de performance et de comparaison du projet.

#### 1. Explications du script (Les 3 points clés pour toi)

1.
**Préparation Rigoureuse (Data Pipeline) :** * On charge les 1797 images de  pixels.


* On respecte ton rapport en découpant le jeu de données : 1437 images pour l'entraînement (80%) et le reste pour le test.


* On applique la **Standardisation** . C'est crucial car, comme le note ton rapport, cela évite les erreurs de calcul (overflow) lors de l'utilisation des fonctions exponentielles dans le Softmax.




2. **Double Entraînement (From Scratch vs Scikit-Learn) :**
* Le script entraîne d'abord ton modèle "maison" en passant les données à `LogitScratch`, qui elle-même appelle la descente de gradient du professeur.


* Ensuite, il entraîne le modèle officiel de Scikit-Learn avec les mêmes données. Cela permet une comparaison "toutes choses égales par ailleurs".




3. **Métriques et Validation :** * Pour l'analyse, on utilise deux outils vus en TD : l'**Accuracy** (pourcentage de bonnes réponses) et le **Log Loss** (entropie croisée).
* Cela permet de vérifier si ton implémentation arrive à la même "certitude" mathématique que l'outil professionnel.


---

### Ouverture Critique

1. **Sur la convergence :** Tu remarqueras peut-être que le modèle "From Scratch" demande un nombre d'itérations plus élevé ou un `learning_rate` plus précis que Scikit-Learn pour atteindre la même précision. C'est normal : Scikit-Learn utilise des algorithmes d'optimisation plus complexes (comme L-BFGS) que la descente de gradient simple vue en cours.


2.
**Analyse des coefficients :** Le projet demande d'interpréter les coefficients appris. Avec cette structure, tu peux maintenant accéder à `model_home.theta`. Chaque ligne de cette matrice représente un chiffre (0 à 9) et chaque colonne un pixel.