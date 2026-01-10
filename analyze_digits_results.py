import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from softmax_regression_from_scratch import SoftmaxRegressionGD, add_bias

TEST_SIZE = 0.2
SEED = 0

LR = 0.1
EPOCHS = 5000
TOL = 1e-6
L2 = 0.0

def per_class_acc(cm):
    d = cm.sum(axis=1)
    d[d == 0] = 1
    return np.diag(cm) / d

def report(name, y_true, y_pred, proba):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(10))
    print(f"\n--- {name} ---")
    print(f"accuracy  = {accuracy_score(y_true, y_pred):.4f}")
    print(f"log_loss  = {log_loss(y_true, proba, labels=np.arange(10)):.4f}")
    print("per-class accuracy =", np.round(per_class_acc(cm), 4))
    print("\nMatrice de confusion:\n", cm)
    print("\nClassification report:\n", classification_report(y_true, y_pred, digits=4))
    return cm

def main():
    digits = load_digits()
    X = digits.data.astype(float)
    y = digits.target.astype(int)
    images = digits.images

    X_tr, X_te, y_tr, y_te, img_tr, img_te = train_test_split(
        X, y, images, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    X_tr_b = add_bias(X_tr)
    X_te_b = add_bias(X_te)

    scratch = SoftmaxRegressionGD(lr=LR, epochs=EPOCHS, tol=TOL, l2=L2, seed=SEED, verbose=False)
    scratch.fit(X_tr_b, y_tr, C=10)
    p_s = scratch.predict_proba(X_te_b)
    y_s = p_s.argmax(axis=1)

    sk = LogisticRegression(max_iter=20000)
    sk.fit(X_tr, y_tr)
    p_k = sk.predict_proba(X_te)
    y_k = p_k.argmax(axis=1)

    cm_s = report("from_scratch_softmax", y_te, y_s, p_s)
    cm_k = report("sklearn_logreg", y_te, y_k, p_k)

    plt.figure()
    plt.imshow(cm_s)
    plt.title("Confusion scratch")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))

    plt.figure()
    plt.imshow(cm_k)
    plt.title("Confusion sklearn")
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))

    idx = np.where(y_te != y_s)[0][:16]
    if len(idx):
        n = len(idx)
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))
        plt.figure()
        for k, i in enumerate(idx, 1):
            plt.subplot(rows, cols, k)
            plt.imshow(img_te[i], cmap="gray")
            plt.axis("off")
            plt.title(f"{y_te[i]}->{y_s[i]}")
        plt.suptitle("Erreurs scratch")

    w_s = scratch.theta[1:].T
    w_k = sk.coef_

    plt.figure(figsize=(10, 4))
    for c in range(10):
        plt.subplot(2, 5, c + 1)
        plt.imshow(w_s[c].reshape(8, 8))
        plt.axis("off")
        plt.title(str(c))
    plt.suptitle("Poids scratch")

    plt.figure(figsize=(10, 4))
    for c in range(10):
        plt.subplot(2, 5, c + 1)
        plt.imshow(w_k[c].reshape(8, 8))
        plt.axis("off")
        plt.title(str(c))
    plt.suptitle("Poids sklearn")

    plt.show()

if __name__ == "__main__":
    main()