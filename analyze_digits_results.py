import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report


def per_class_acc(cm):
    # on calcule la precision pour chaque chiffre
    d = cm.sum(axis=1)
    d[d == 0] = 1
    return np.diag(cm) / d


def report(name, y_true, y_pred, proba):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    print(f"\n--- DIAGNOSTIC DÉTAILLÉ : {name} ---")
    print(f"Accuracy globale = {accuracy_score(y_true, y_pred):.4f}")
    print(f"Log Loss (Erreur) = {log_loss(y_true, proba, labels=np.arange(10)):.4f}")
    print("Précision par chiffre (0 à 9) :")
    print(np.round(per_class_acc(cm), 4))
    print("\nClassification report :\n", classification_report(y_true, y_pred, digits=4))
    return cm


def show_examples(title, imgs, y_true, y_pred, idx, max_n=12):
    #on affiche des exemples de digits bien identifiés et mal identifiés
    idx = idx[:max_n]
    if len(idx) == 0:
        return
    n = len(idx)
    cols = 4
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(10, 2 * rows))
    for k, i in enumerate(idx, 1):
        plt.subplot(rows, cols, k)
        plt.imshow(imgs[i], cmap="gray")
        plt.axis("off")
        # g = guess (prédiction), r = real (vérité)
        plt.title(f"guess={y_pred[i]}, real={y_true[i]}", fontsize=9)
    plt.suptitle(title)
    plt.tight_layout()