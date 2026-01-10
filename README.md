# Scripts - Projet Digits (régression logistique multi-classe)

Ces scripts ajoutent une partie **pratique** à votre rapport théorique :
- implémentation **from scratch** d’une régression logistique multi-classe (softmax),
- comparaison avec **scikit-learn**,
- éléments d’analyse (erreurs + interprétation des coefficients).

## Fichiers
- `softmax_regression_from_scratch.py` : modèle softmax + descente de gradient (full batch).
- `run_digits_experiment.py` : pipeline complet + métriques + comparaison sklearn.
- `analyze_digits_results.py` : figures d’analyse (erreurs, poids 8×8, matrices de confusion).

## Exécuter
Depuis ce dossier :

```bash
python run_digits_experiment.py
python run_digits_experiment.py --lr 0.2 --max-epochs 8000 --grad-tol 1e-7 --save-outputs
python analyze_digits_results.py --save-figs
```

## Correspondance avec le sujet
- **Implémenter** la régression logistique multi-classes + descente de gradient : `softmax_regression_from_scratch.py`
- **Comparer** avec scikit-learn (découpage / métriques) : `run_digits_experiment.py`
- **Analyser** (erreurs / coefficients / influence des paramètres) : `analyze_digits_results.py`
